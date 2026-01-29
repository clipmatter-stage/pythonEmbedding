# Video Transcript Elastic Search API

A secure, production-ready FastAPI service for semantic search over video transcripts with fuzzy matching and typo tolerance.

## Features

- 🔍 **Semantic Search** - AI-powered search using sentence transformers
- 🎯 **Fuzzy Matching** - Typo-tolerant search (e.g., "Muhamad" matches "Muhammad")
- 🔐 **API Key Authentication** - Optional API key protection
- ⚡ **Rate Limiting** - Prevents abuse (configurable)
- 📝 **Input Validation** - Pydantic models for all inputs
- 📊 **Proper Logging** - Structured logging (no print statements)
- 🚀 **Railway Ready** - Configured for Railway deployment

## Setup

### 1. Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Required variables:
```env
QDRANT_URL=https://your-qdrant-instance.cloud.qdrant.io:6333
QDRANT_API_KEY=your-qdrant-api-key

# Optional
API_KEY=your-api-key-for-clients  # Leave empty to disable auth
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60
LOG_LEVEL=INFO
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Locally

```bash
uvicorn embeddings_test:app --reload --port 9000
```

## API Endpoints

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/embed-video` | Embed video transcript | Yes |
| POST | `/search` | Elastic search | Yes |
| POST | `/search-by-title` | Search by video title | Yes |
| POST | `/suggest` | Autocomplete suggestions | Yes |
| GET | `/video/{id}/segments` | Get video segments | Yes |
| DELETE | `/video/{id}/embeddings` | Delete embeddings | Yes |
| GET | `/health` | Health check | No |
| GET | `/stats` | Collection stats | No |

## Authentication

If `API_KEY` is set, include header in requests:

```bash
curl -X POST https://your-api.railway.app/search \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning"}'
```

## Security Best Practices

1. **Never commit `.env`** - It's in `.gitignore`
2. **Use environment variables** - All secrets from env vars
3. **Enable API key** - Set `API_KEY` in production
4. **Configure CORS** - Set `CORS_ORIGINS` for your domains

## Railway Deployment

1. Push to GitHub
2. Connect repo to Railway
3. Add environment variables in Railway dashboard
4. Deploy!

## License

MIT