# AI Meeting Assistant

An agentic meeting assistant that transcribes audio, extracts action items and summaries using LangChain agents, and provides RAG-based question answering over historical meeting notes.

## Features

- **Audio Transcription**: Upload audio files and get transcripts using OpenAI Whisper
- **Smart Extraction**: LangGraph agent extracts summaries, action items, and decisions from transcripts
- **Human-in-the-Loop**: Low-confidence extractions pause for human review and correction
- **RAG Q&A**: Ask questions about past meetings using vector similarity search
- **Offline Evaluation**: Benchmark RAG and extraction accuracy with golden datasets
- **Vector Search**: pgvector-powered semantic search over 500+ indexed meetings

## Tech Stack

- **Backend**: Python 3.11+, FastAPI, LangChain, LangGraph
- **AI**: OpenAI Whisper (transcription), GPT-4o (LLM), text-embedding-3-small
- **Database**: PostgreSQL 16 + pgvector extension
- **Package Manager**: uv (not pip)
- **Testing**: pytest, pytest-asyncio
- **Linting**: ruff, black

## Project Structure

```
ai-meeting-assistant/
├── app/
│   ├── main.py                   # FastAPI app entry point
│   ├── api/routes/               # API endpoints
│   ├── agents/                   # LangGraph agents
│   ├── services/                 # Business logic layer
│   ├── models/                   # SQLAlchemy & Pydantic models
│   ├── core/                     # Config, database, exceptions
│   └── utils/                    # Logging utilities
├── tests/                        # Test suite
├── scripts/                      # Utility scripts
├── golden_sets/                  # Evaluation datasets
├── alembic/                      # Database migrations
└── docker-compose.yml            # Local PostgreSQL setup
```

## Local Development Setup

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- OpenAI API key

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Aditya003-B/ai-meeting-assistant.git
cd ai-meeting-assistant
```

2. **Install uv package manager**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. **Install dependencies**
```bash
uv sync
```

4. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

5. **Start PostgreSQL + pgvector**
```bash
docker-compose up -d
```

6. **Run database migrations**
```bash
uv run alembic upgrade head
```

7. **Seed the database with synthetic meetings**
```bash
uv run python scripts/seed_meetings.py
```

8. **Start the development server**
```bash
uv run uvicorn app.main:app --reload --port 8000
```

9. **Visit the API documentation**
```bash
open http://localhost:8000/docs
```

## API Endpoints

### Transcription
- `POST /transcribe` - Upload audio file and get transcript

### Meetings
- `POST /meetings` - Create meeting from transcript (runs extraction agent)
- `GET /meetings` - List all meetings (paginated, searchable)
- `GET /meetings/{id}` - Get specific meeting details
- `PATCH /meetings/{id}/confirm` - Confirm/correct extracted information

### Query
- `POST /query` - Ask questions about meetings (RAG)
- `POST /eval/run` - Run offline evaluation pipeline

### Health
- `GET /` - Health check
- `GET /health` - Detailed health information

## Usage Examples

### 1. Transcribe Audio
```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@meeting.mp3"
```

### 2. Create Meeting
```bash
curl -X POST http://localhost:8000/meetings \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Sprint Planning Q2",
    "transcript": "John will handle the API migration by end of month..."
  }'
```

### 3. Query Meetings
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What action items were assigned to John?",
    "top_k": 5
  }'
```

## Testing

Run the test suite:
```bash
uv run pytest
```

Run with coverage:
```bash
uv run pytest --cov=app --cov-report=html
```

## Code Quality

Format code:
```bash
uv run black app tests scripts
```

Lint code:
```bash
uv run ruff check app tests scripts
```

## Deployment

**Deploy to production for FREE using Render + Neon.**

See **[DEPLOY.md](DEPLOY.md)** for complete deployment guide.

### Quick Deploy

1. **Database:** Create Neon PostgreSQL database with pgvector
2. **Backend:** Deploy to Render using Docker
3. **Configure:** Set environment variables in Render dashboard
4. **Done:** Your API is live at `https://your-app.onrender.com`

**Free Tier:**
- Render: 750 hours/month, 512 MB RAM (spins down after 15 min)
- Neon: 512 MB storage, always-on database

**Auto-deploy:** Push to `main` branch triggers automatic deployment on Render.

## Environment Variables

Required environment variables (see `.env.example`):

```
OPENAI_API_KEY=sk-your-key-here
DATABASE_URL=postgresql+asyncpg://user:pass@host/db
ENVIRONMENT=development
LOG_LEVEL=INFO
MAX_AUDIO_SIZE_MB=25
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o
CONFIDENCE_THRESHOLD=0.75
```

## Architecture

### LangGraph Extraction Agent

The extraction agent uses a multi-node workflow:
1. **Summarize** - Generate 3-5 sentence summary
2. **Extract Actions** - Identify action items with owner and deadline
3. **Extract Decisions** - Identify key decisions made
4. **Score Confidence** - Calculate confidence score (0-1)
5. **Human Review Gate** - If confidence < 0.75, require human confirmation

### RAG Pipeline

**Indexing:**
1. Combine title + summary + transcript
2. Generate embedding using text-embedding-3-small
3. Store in pgvector with meeting metadata

**Retrieval:**
1. Embed user question
2. Cosine similarity search in pgvector
3. Retrieve top-k most relevant meetings
4. GPT-4o generates answer with source attribution

## License

MIT

## Author

Aditya Bhardwaj (b.aditya26@gmail.com)
