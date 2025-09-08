import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import multer from 'multer';
import fs from 'fs';
import path from 'path';
import mime from 'mime-types';
import PDFDocument from 'pdfkit';
import pkg from '@deepgram/sdk';
import { GoogleGenAI, createUserContent, Type } from '@google/genai';

const { Deepgram } = pkg;

const app = express();
app.use(cors());
app.use(express.json({ limit: '5mb' }));
app.use(express.static('public'));

const UPLOAD_DIR = path.resolve('./uploads');
if (!fs.existsSync(UPLOAD_DIR)) fs.mkdirSync(UPLOAD_DIR, { recursive: true });

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, UPLOAD_DIR),
  filename: (req, file, cb) => cb(null, Date.now() + '-' + file.originalname.replace(/\s+/g,'_'))
});
const upload = multer({ storage });

// Envs
const GEMINI_API_KEY = process.env.GEMINI_API_KEY || '';
const GEMINI_MODEL = process.env.GEMINI_MODEL || 'gemini-2.5-flash';
const DEEPGRAM_API_KEY = process.env.DEEPGRAM_API_KEY || '';
const DEEPGRAM_LANGUAGE = process.env.DEEPGRAM_LANGUAGE || 'es';

if (!GEMINI_API_KEY) console.warn('⚠️ Falta GEMINI_API_KEY en el entorno');
if (!DEEPGRAM_API_KEY) console.warn('⚠️ Falta DEEPGRAM_API_KEY en el entorno');

// Clients
const ai = new GoogleGenAI({ apiKey: GEMINI_API_KEY });
const deepgram = new Deepgram(DEEPGRAM_API_KEY);

// Util: format time
function fmtTime(secs){
  if (typeof secs !== 'number') return '0.00';
  return secs.toFixed(2) + 's';
}

// Endpoint principal
app.post('/api/process', upload.single('file'), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'No se recibió archivo' });
  const filePath = req.file.path;
  const mt = mime.lookup(filePath) || 'audio/mpeg';
  try {
    // 1) Diarize + Transcribe via Deepgram (pre-recorded)
    const audioBuffer = fs.readFileSync(filePath);
    let dgResp = null;
    try {
      dgResp = await deepgram.transcription.preRecorded(
        { buffer: audioBuffer, mimetype: mt },
        {
          punctuate: true,
          diarize: true,
          utterances: true,
          language: DEEPGRAM_LANGUAGE
        }
      );
    } catch (e) {
      console.warn('Deepgram error:', e?.message || e);
      // fallback: set dgResp to null and continue
    }

    // Extract transcript and utterances
    let transcript = '';
    let utterances = [];
    try {
      transcript = dgResp?.results?.channels?.[0]?.alternatives?.[0]?.transcript || '';
      utterances = dgResp?.results?.utterances || [];
    } catch (e) {
      transcript = transcript || '';
      utterances = utterances || [];
    }

    // Build diarized text for Gemini prompt
    let diarized_text = '';
    if (utterances && utterances.length > 0) {
      utterances.forEach((u, i) => {
        const speaker = u.speaker || u.speaker_label || ('SPEAKER_' + (u.speaker ? u.speaker : i+1));
        diarized_text += `(${fmtTime(u.start)} - ${fmtTime(u.end)}) ${speaker}: ${u.transcript}\n`;
      });
    } else {
      diarized_text = transcript || 'No se pudo transcribir con Deepgram.';
    }

    // 2) Prepare Structured Output schema for Gemini (JSON)
    const responseSchema = {
      type: Type.OBJECT,
      properties: {
        titulo: { type: Type.STRING },
        fecha: { type: Type.STRING },
        asistentes: { type: Type.ARRAY, items: { type: Type.OBJECT, properties: { nombre: { type: Type.STRING }, rol: { type: Type.STRING } } } },
        agenda: { type: Type.ARRAY, items: { type: Type.OBJECT, properties: { tema: { type: Type.STRING }, resumen: { type: Type.STRING } } } },
        decisiones: { type: Type.ARRAY, items: { type: Type.STRING } },
        acuerdos: { type: Type.ARRAY, items: { type: Type.OBJECT, properties: { responsable: { type: Type.STRING }, tarea: { type: Type.STRING }, fecha_compromiso: { type: Type.STRING } } } }
      }
    };

    const promptIntro = [
      `A continuación tienes la transcripción diarizada (etiquetada por hablante) de una reunión:`,
      `\n\n---\n\n`,
      diarized_text,
      `\n\n---\n\n`,
      `Genera una MINUTA estructurada (JSON) con las propiedades: titulo, fecha (ISO-8601 si se menciona), asistentes (nombre, rol si se puede inferir), agenda (tema y resumen breve), decisiones (lista) y acuerdos (responsable, tarea, fecha_compromiso).`,
      `Sé fiel al contenido: no inventes asistentes ni decisiones. Si falta información, indica "No mencionado".`
    ];

    // 3) Call Gemini for structured JSON (minuta)
    const minutaResp = await ai.models.generateContent({
      model: GEMINI_MODEL,
      contents: createUserContent(promptIntro),
      config: {
        responseMimeType: 'application/json',
        responseSchema
      }
    });

    // 4) Call Gemini for executive summary in Markdown
    const resumenParts = [
      diarized_text,
      "Redacta una minuta ejecutiva en Markdown con:",
      "- Resumen (3-5 bullets),",
      "- Decisiones clave,",
      "- Lista de acuerdos con responsables y fechas,",
      "- Riesgos o bloqueos (si existen).",
      "Sé conciso y claro en español."
    ];
    const resumenResp = await ai.models.generateContent({
      model: GEMINI_MODEL,
      contents: createUserContent(resumenParts)
    });

    // Try to parse structured JSON (minuta)
    let minuta = null;
    try {
      // The SDK may return .text or .output; try available properties
      if (minutaResp?.text) {
        minuta = JSON.parse(minutaResp.text);
      } else if (minutaResp?.output && Array.isArray(minutaResp.output) && minutaResp.output[0]?.content?.[0]?.text) {
        minuta = JSON.parse(minutaResp.output[0].content[0].text);
      } else {
        minuta = minutaResp;
      }
    } catch (e) {
      minuta = { raw: minutaResp?.text || JSON.stringify(minutaResp) };
    }

    const resumen_md = resumenResp?.text || (resumenResp?.output?.[0]?.content?.[0]?.text) || '';

    // 5) Generate PDF from resumen_md (simple plain-text rendering)
    const pdfName = path.basename(filePath) + '.pdf';
    const pdfPath = path.join(UPLOAD_DIR, pdfName);

    await new Promise((resolve, reject) => {
      const doc = new PDFDocument({ margin: 50 });
      const stream = fs.createWriteStream(pdfPath);
      doc.pipe(stream);
      // Header
      doc.fontSize(18).text('Minuta ejecutiva', { align: 'center' });
      doc.moveDown();
      // Add basic metadata
      doc.fontSize(10).text(`Generado: ${new Date().toISOString()}`);
      doc.moveDown();
      // Write markdown lines as plain text
      const lines = (resumen_md || '').split('\n');
      doc.fontSize(12);
      lines.forEach(line => {
        doc.text(line, { paragraphGap: 2 });
      });
      doc.end();
      stream.on('finish', resolve);
      stream.on('error', reject);
    });

    // 6) Responder al cliente
    const pdf_url = `/download/${encodeURIComponent(pdfName)}`;
    res.json({
      model: GEMINI_MODEL,
      minuta,
      resumen_md,
      pdf_url,
      diarization: { transcript, utterances_count: utterances.length || 0 }
    });

  } catch (err) {
    console.error('Processing error:', err);
    res.status(500).json({ error: 'Fallo procesando el audio', details: err?.message || String(err) });
  } finally {
    // cleanup original uploaded audio (optional)
    try { fs.unlinkSync(filePath); } catch (e) { /* ignore */ }
  }
});

// Route for downloading PDF
app.get('/download/:name', (req, res) => {
  const name = req.params.name;
  const p = path.join(UPLOAD_DIR, name);
  if (!fs.existsSync(p)) return res.status(404).send('Not found');
  res.download(p, name, (err) => {
    if (err) console.warn('Download error', err);
  });
});

app.get('/api/health', (_req, res) => res.json({ ok: true, model: GEMINI_MODEL }));

const PORT = process.env.PORT || 10000;
app.listen(PORT, () => console.log(`✅ MinutAI upgraded escuchando en http://0.0.0.0:${PORT}`));
