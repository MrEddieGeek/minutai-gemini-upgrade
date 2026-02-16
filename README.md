# MinutAI

Transcripción con diarización (Deepgram) + minuta estructurada (OpenAI) + exportación a PDF.

## Funcionalidades

- **Transcripción y diarización** con Deepgram (pre-recorded, utterances)
- **Minuta estructurada** (JSON) y resumen ejecutivo (Markdown) generados con OpenAI
- **Exportación a PDF** con formato Markdown (encabezados, negritas, listas con viñetas)
- **Progreso en tiempo real** via Server-Sent Events (SSE)
- **Selector de idioma** (español, inglés, portugués, francés, alemán, italiano)
- **Validación de archivo** (máximo 100 MB, muestra nombre y tamaño)
- **Mapeo de hablantes** — renombra SPEAKER_0, SPEAKER_1, etc. con nombres reales
- **Minuta editable** — edita el Markdown y regenera el PDF desde el navegador

## Requisitos

- Node 20+
- Cuenta OpenAI (API key)
- Cuenta Deepgram (API key)

## Uso local

1. Copia `.env.example` a `.env` y añade tus claves:
   ```
   OPENAI_API_KEY=...
   DEEPGRAM_API_KEY=...
   ```
2. `npm install`
3. `npm run dev`
4. Abre `http://localhost:10000` y sube un audio.

## Despliegue

Compatible con Render u otro servicio Docker-compatible. Ver `render.yaml` y `Dockerfile`.
