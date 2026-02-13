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

### Secure Secret Management

This project uses a custom Dockerfile (not Nixpacks) to avoid embedding secrets in the Docker image.

**⚠️ IMPORTANT: Secrets are injected at runtime by Railway, NOT baked into the Docker image.**

### Deployment Steps

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Add secure Dockerfile"
   git push origin main
   ```

2. **Connect to Railway**
   - Go to [railway.app](https://railway.app)
   - Create new project from GitHub repo
   - Railway will detect the Dockerfile automatically

3. **Configure Environment Variables in Railway Dashboard**
   
   Go to your project → Variables tab and add:
   
   | Variable | Value | Required |
   |----------|-------|----------|
   | `QDRANT_URL` | Your Qdrant instance URL | ✅ Yes |
   | `QDRANT_API_KEY` | Your Qdrant API key | ✅ Yes |
   | `API_KEY` | Your custom API key | ⚠️ Recommended |
   | `PORT` | Auto-set by Railway | ✅ Auto |
   | `LOG_LEVEL` | INFO | Optional |
   | `RATE_LIMIT_REQUESTS` | 100 | Optional |
   | `RATE_LIMIT_WINDOW` | 60 | Optional |
   | `CORS_ORIGINS` | * or your domains | Optional |

4. **Deploy**
   - Railway will automatically build and deploy
   - Check logs for any errors
   - Test with: `https://your-app.railway.app/health`

### Why Custom Dockerfile?

The custom Dockerfile:
- ✅ Does NOT use `ARG` or `ENV` for secrets (Docker security best practice)
- ✅ Expects secrets from Railway environment variables at runtime
- ✅ Uses multi-stage builds for smaller image size
- ✅ Runs as non-root user for security
- ✅ Includes health checks

### Troubleshooting

**Error: "Missing required environment variables"**
- Ensure `QDRANT_URL` and `QDRANT_API_KEY` are set in Railway Variables tab

**Error: "401 Unauthorized"**
- If `API_KEY` is set, include `X-API-Key` header in requests
- To disable auth, remove `API_KEY` variable

**Build fails**
- Check Railway build logs
- Ensure `requirements.txt` is up to date
- Verify Dockerfile syntax

### Testing Your Deployment

```bash
# Health check
curl https://your-app.railway.app/health

# Search (with API key)
curl -X POST https://your-app.railway.app/search \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning", "top_k": 5}'
```

## License

MIT