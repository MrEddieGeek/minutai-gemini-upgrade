import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import multer from 'multer';
import fs from 'fs';
import path from 'path';
import mime from 'mime-types';
import PDFDocument from 'pdfkit';
import pkg from '@deepgram/sdk';
import OpenAI from 'openai';
import markdownIt from 'markdown-it';

const { Deepgram } = pkg;
const md = markdownIt();

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
const upload = multer({
  storage,
  limits: { fileSize: 100 * 1024 * 1024 } // 100 MB
});

// Envs
const OPENAI_API_KEY = process.env.OPENAI_API_KEY || '';
const OPENAI_MODEL = process.env.OPENAI_MODEL || 'gpt-4.1-mini';
const DEEPGRAM_API_KEY = process.env.DEEPGRAM_API_KEY || '';
const DEEPGRAM_LANGUAGE = process.env.DEEPGRAM_LANGUAGE || 'es';

if (!OPENAI_API_KEY) console.warn('⚠️ Falta OPENAI_API_KEY en el entorno');
if (!DEEPGRAM_API_KEY) console.warn('⚠️ Falta DEEPGRAM_API_KEY en el entorno');

// Clients
const openai = new OpenAI({ apiKey: OPENAI_API_KEY });
const deepgram = new Deepgram(DEEPGRAM_API_KEY);

// Util: format time
function fmtTime(secs){
  if (typeof secs !== 'number') return '0.00';
  return secs.toFixed(2) + 's';
}

// Util: language name helper
function getLanguageName(code) {
  const map = {
    es: 'español', en: 'English', pt: 'português',
    fr: 'français', de: 'Deutsch', it: 'italiano'
  };
  return map[code] || map.es;
}

// Util: render markdown to PDF using markdown-it tokens + PDFKit
function renderMarkdownToPDF(doc, markdown) {
  const tokens = md.parse(markdown, {});
  let isBold = false;
  let isBullet = false;
  let headingLevel = 0;

  for (let i = 0; i < tokens.length; i++) {
    const token = tokens[i];

    if (token.type === 'heading_open') {
      headingLevel = parseInt(token.tag.replace('h', ''), 10);
      const sizes = { 1: 18, 2: 16, 3: 14, 4: 13, 5: 12, 6: 11 };
      doc.fontSize(sizes[headingLevel] || 14).font('Helvetica-Bold');
      continue;
    }

    if (token.type === 'heading_close') {
      doc.moveDown(0.3);
      headingLevel = 0;
      doc.fontSize(11).font('Helvetica');
      continue;
    }

    if (token.type === 'bullet_list_open') {
      isBullet = true;
      continue;
    }
    if (token.type === 'bullet_list_close') {
      isBullet = false;
      doc.moveDown(0.3);
      continue;
    }

    if (token.type === 'list_item_open') continue;
    if (token.type === 'list_item_close') continue;

    if (token.type === 'paragraph_open') continue;
    if (token.type === 'paragraph_close') {
      if (!isBullet) doc.moveDown(0.4);
      continue;
    }

    if (token.type === 'inline' && token.children) {
      const indent = isBullet ? 20 : 0;
      if (isBullet) {
        doc.text('', { indent: 0 }); // flush
      }
      let lineText = '';
      // We'll build segments and render them
      const segments = [];
      for (const child of token.children) {
        if (child.type === 'strong_open') {
          if (lineText) segments.push({ text: lineText, bold: false });
          lineText = '';
          isBold = true;
          continue;
        }
        if (child.type === 'strong_close') {
          if (lineText) segments.push({ text: lineText, bold: true });
          lineText = '';
          isBold = false;
          continue;
        }
        if (child.type === 'em_open' || child.type === 'em_close') continue;
        if (child.type === 'softbreak') {
          lineText += ' ';
          continue;
        }
        if (child.type === 'text' || child.type === 'code_inline') {
          lineText += child.content;
        }
      }
      if (lineText) segments.push({ text: lineText, bold: isBold });

      // Render segments
      const prefix = isBullet ? '  \u2022  ' : '';
      const xIndent = isBullet ? indent : 0;

      if (segments.length === 0) continue;

      // Simple case: single segment
      if (segments.length === 1) {
        const s = segments[0];
        doc.font(s.bold ? 'Helvetica-Bold' : 'Helvetica');
        if (headingLevel === 0) doc.fontSize(11);
        doc.text(prefix + s.text, { indent: xIndent, lineGap: 2 });
      } else {
        // Multiple segments: use doc.text with continued
        for (let si = 0; si < segments.length; si++) {
          const s = segments[si];
          doc.font(s.bold ? 'Helvetica-Bold' : 'Helvetica');
          if (headingLevel === 0) doc.fontSize(11);
          const txt = si === 0 ? prefix + s.text : s.text;
          const opts = si < segments.length - 1
            ? { continued: true, indent: si === 0 ? xIndent : 0, lineGap: 2 }
            : { indent: 0, lineGap: 2 };
          doc.text(txt, opts);
        }
      }
      doc.font('Helvetica');
      continue;
    }

    // Fallback for hr, code blocks, etc.
    if (token.type === 'hr') {
      doc.moveDown(0.5);
      doc.moveTo(doc.x, doc.y).lineTo(doc.x + 450, doc.y).stroke('#cccccc');
      doc.moveDown(0.5);
      continue;
    }

    if (token.type === 'fence' || token.type === 'code_block') {
      doc.fontSize(10).font('Courier').text(token.content, { indent: 10 });
      doc.font('Helvetica').fontSize(11);
      doc.moveDown(0.3);
      continue;
    }
  }
}

// Helper: send SSE event
function sendSSE(res, event, data) {
  res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
}

// Endpoint principal — now uses SSE for real-time progress
app.post('/api/process', (req, res, next) => {
  upload.single('file')(req, res, (err) => {
    if (err) {
      if (err.code === 'LIMIT_FILE_SIZE') {
        return res.status(400).json({ error: 'El archivo excede el límite de 100 MB' });
      }
      return res.status(400).json({ error: err.message });
    }
    next();
  });
}, async (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'No se recibió archivo' });

  // Set up SSE
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive'
  });

  const filePath = req.file.path;
  const mt = mime.lookup(filePath) || 'audio/mpeg';
  const language = req.body.language || DEEPGRAM_LANGUAGE;
  const langName = getLanguageName(language);

  try {
    // 1) Diarize + Transcribe via Deepgram
    sendSSE(res, 'progress', { step: 'Transcribiendo audio con Deepgram...' });

    const audioBuffer = fs.readFileSync(filePath);
    let dgResp = null;
    try {
      dgResp = await deepgram.transcription.preRecorded(
        { buffer: audioBuffer, mimetype: mt },
        {
          punctuate: true,
          diarize: true,
          utterances: true,
          language: language
        }
      );
    } catch (e) {
      console.warn('Deepgram error:', e?.message || e);
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

    // Build diarized text for prompt
    let diarized_text = '';
    if (utterances && utterances.length > 0) {
      utterances.forEach((u, i) => {
        const speaker = u.speaker !== undefined ? `SPEAKER_${u.speaker}` : ('SPEAKER_' + (i+1));
        diarized_text += `(${fmtTime(u.start)} - ${fmtTime(u.end)}) ${speaker}: ${u.transcript}\n`;
      });
    } else {
      diarized_text = transcript || 'No se pudo transcribir con Deepgram.';
    }

    sendSSE(res, 'progress', { step: `Transcripción completa. ${utterances.length} utterances detectadas.` });

    // 2) Call OpenAI for structured JSON (minuta)
    sendSSE(res, 'progress', { step: 'Generando minuta estructurada (JSON)...' });

    const minutaSystemPrompt = language === 'es'
      ? 'Eres un asistente que genera minutas estructuradas de reuniones en formato JSON. Responde SOLO con JSON válido.'
      : `Eres un asistente que genera minutas estructuradas de reuniones en formato JSON. Responde SOLO con JSON válido. Responde en ${langName}.`;

    const minutaUserPrompt = language === 'es'
      ? [
          'A continuación tienes la transcripción diarizada (etiquetada por hablante) de una reunión:',
          '\n\n---\n\n',
          diarized_text,
          '\n\n---\n\n',
          'Genera una MINUTA estructurada (JSON) con las propiedades: titulo, fecha (ISO-8601 si se menciona), asistentes (nombre, rol si se puede inferir), agenda (tema y resumen breve), decisiones (lista) y acuerdos (responsable, tarea, fecha_compromiso).',
          'Sé fiel al contenido: no inventes asistentes ni decisiones. Si falta información, indica "No mencionado".'
        ].join('')
      : [
          'Below is the diarized transcript (labeled by speaker) of a meeting:',
          '\n\n---\n\n',
          diarized_text,
          '\n\n---\n\n',
          `Generate a structured MINUTES (JSON) with properties: title, date (ISO-8601 if mentioned), attendees (name, role if inferable), agenda (topic and brief summary), decisions (list) and agreements (responsible, task, deadline).`,
          `Be faithful to the content: do not invent attendees or decisions. If information is missing, indicate "Not mentioned". Respond in ${langName}.`
        ].join('');

    const minutaResp = await openai.chat.completions.create({
      model: OPENAI_MODEL,
      response_format: { type: 'json_object' },
      messages: [
        { role: 'system', content: minutaSystemPrompt },
        { role: 'user', content: minutaUserPrompt }
      ]
    });

    sendSSE(res, 'progress', { step: 'Minuta JSON generada. Generando resumen ejecutivo...' });

    // 3) Call OpenAI for executive summary in Markdown
    const resumenSystemPrompt = language === 'es'
      ? 'Eres un asistente que redacta minutas ejecutivas en Markdown. Sé conciso y claro en español.'
      : `Eres un asistente que redacta minutas ejecutivas en Markdown. Sé conciso y claro. Responde en ${langName}.`;

    const resumenUserPrompt = language === 'es'
      ? [
          diarized_text,
          '\n\nRedacta una minuta ejecutiva en Markdown con:',
          '\n- Resumen (3-5 bullets),',
          '\n- Decisiones clave,',
          '\n- Lista de acuerdos con responsables y fechas,',
          '\n- Riesgos o bloqueos (si existen).'
        ].join('')
      : [
          diarized_text,
          `\n\nWrite an executive summary in Markdown in ${langName} with:`,
          '\n- Summary (3-5 bullets),',
          '\n- Key decisions,',
          '\n- List of agreements with responsible parties and dates,',
          '\n- Risks or blockers (if any).'
        ].join('');

    const resumenResp = await openai.chat.completions.create({
      model: OPENAI_MODEL,
      messages: [
        { role: 'system', content: resumenSystemPrompt },
        { role: 'user', content: resumenUserPrompt }
      ]
    });

    // Parse structured JSON (minuta)
    let minuta = null;
    try {
      minuta = JSON.parse(minutaResp.choices[0].message.content);
    } catch (e) {
      minuta = { raw: minutaResp.choices[0].message.content };
    }

    const resumen_md = resumenResp.choices[0].message.content || '';

    sendSSE(res, 'progress', { step: 'Generando PDF...' });

    // 4) Generate PDF from resumen_md using markdown renderer
    const pdfName = path.basename(filePath) + '.pdf';
    const pdfPath = path.join(UPLOAD_DIR, pdfName);

    await new Promise((resolve, reject) => {
      const doc = new PDFDocument({ margin: 50 });
      const stream = fs.createWriteStream(pdfPath);
      doc.pipe(stream);
      doc.fontSize(20).font('Helvetica-Bold').text('Minuta ejecutiva', { align: 'center' });
      doc.moveDown(0.3);
      doc.fontSize(10).font('Helvetica').text(`Generado: ${new Date().toISOString()}`, { align: 'center' });
      doc.moveDown(1);

      renderMarkdownToPDF(doc, resumen_md);

      doc.end();
      stream.on('finish', resolve);
      stream.on('error', reject);
    });

    // 5) Send complete event
    const pdf_url = `/download/${encodeURIComponent(pdfName)}`;
    sendSSE(res, 'complete', {
      model: OPENAI_MODEL,
      minuta,
      resumen_md,
      pdf_url,
      utterances: utterances.map(u => ({
        speaker: u.speaker !== undefined ? `SPEAKER_${u.speaker}` : `SPEAKER_0`,
        start: u.start,
        end: u.end,
        transcript: u.transcript
      })),
      diarization: { transcript, utterances_count: utterances.length || 0 }
    });

    res.end();

  } catch (err) {
    console.error('Processing error:', err);
    sendSSE(res, 'error', { error: 'Fallo procesando el audio', details: err?.message || String(err) });
    res.end();
  } finally {
    try { fs.unlinkSync(filePath); } catch (e) { /* ignore */ }
  }
});

// Endpoint: regenerate PDF from edited markdown
app.post('/api/regenerate-pdf', async (req, res) => {
  const { markdown } = req.body;
  if (!markdown || typeof markdown !== 'string') {
    return res.status(400).json({ error: 'Se requiere el campo "markdown"' });
  }

  try {
    const pdfName = `regen-${Date.now()}.pdf`;
    const pdfPath = path.join(UPLOAD_DIR, pdfName);

    await new Promise((resolve, reject) => {
      const doc = new PDFDocument({ margin: 50 });
      const stream = fs.createWriteStream(pdfPath);
      doc.pipe(stream);
      doc.fontSize(20).font('Helvetica-Bold').text('Minuta ejecutiva', { align: 'center' });
      doc.moveDown(0.3);
      doc.fontSize(10).font('Helvetica').text(`Generado: ${new Date().toISOString()}`, { align: 'center' });
      doc.moveDown(1);

      renderMarkdownToPDF(doc, markdown);

      doc.end();
      stream.on('finish', resolve);
      stream.on('error', reject);
    });

    const pdf_url = `/download/${encodeURIComponent(pdfName)}`;
    res.json({ pdf_url });
  } catch (err) {
    console.error('PDF regeneration error:', err);
    res.status(500).json({ error: 'Fallo generando el PDF', details: err?.message || String(err) });
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

app.get('/api/health', (_req, res) => res.json({ ok: true, model: OPENAI_MODEL }));

const PORT = process.env.PORT || 10000;
app.listen(PORT, () => console.log(`✅ MinutAI escuchando en http://0.0.0.0:${PORT}`));
