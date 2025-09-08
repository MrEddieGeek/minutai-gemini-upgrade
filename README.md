# MinutAI · Gemini (Upgrade: Deepgram diarization + PDF export)

Este repo es una versión **mejorada** de tu proyecto MinutAI (Gemini). Añade:
- **Diarización** y transcripción con **Deepgram** (pre-recorded, utterances).
- **Resumen / Minuta** con **Gemini** (JSON estructurado + Markdown).
- **Export a PDF** de la minuta (descargable desde la UI).

## Requisitos
- Cuenta Gemini (API key)
- Cuenta Deepgram (API key)
- Render o cualquier servicio Docker-compatible para desplegar
- Node 20 (si pruebas localmente)

## Cómo usar local (rápido)
1. Copia `.env.example` a `.env` y añade tus claves.
2. `npm install`
3. `npm run dev`
4. Abre `http://localhost:10000` y sube un audio.
