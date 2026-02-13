# Railway Deployment Fix - PORT Variable Issue

## ✅ What Was Fixed:

The `$PORT` environment variable wasn't being expanded properly, causing uvicorn to fail with "Invalid value for '--port'". 

### Changes Made:

1. **Created `start.sh`** - Bash script that properly handles PORT environment variable
2. **Updated Dockerfile** - Uses the startup script instead of inline CMD
3. **Updated railway.toml** - Removed conflicting startCommand
4. **Updated Procfile** - Now uses start.sh (for non-Docker deployments)

## 🧪 Test Locally (Optional):

### Test with Docker:
```bash
# Build the image
docker build -t video-search .

# Test with PORT variable
docker run -p 9000:9000 -e PORT=9000 -e QDRANT_URL=your_url -e QDRANT_API_KEY=your_key video-search

# Check health
curl http://localhost:9000/health
```

### Test without Docker:
```bash
# Set environment variables
export PORT=9000
export QDRANT_URL=your_qdrant_url
export QDRANT_API_KEY=your_qdrant_key

# Run startup script
bash start.sh
```

## 🚀 Deploy to Railway:

```bash
git add .
git commit -m "Fix PORT variable expansion issue"
git push origin main
```

Railway will:
1. Detect the Dockerfile
2. Build the container
3. Inject PORT environment variable at runtime
4. Run `./start.sh` which properly expands $PORT
5. Start uvicorn on the correct port

## 🔍 Verify Deployment:

Check Railway logs for:
```
Starting uvicorn on port XXXX...
INFO:     Started server process [X]
INFO:     Uvicorn running on http://0.0.0.0:XXXX
```

Test your endpoint:
```bash
curl https://your-app.railway.app/health
```

## 📋 Environment Variables to Set in Railway:

Make sure these are set in Railway Dashboard → Variables:

- ✅ `QDRANT_URL` - Your Qdrant instance URL
- ✅ `QDRANT_API_KEY` - Your Qdrant API key  
- ✅ `API_KEY` - Your custom API key (optional but recommended)
- ⚙️ `PORT` - Auto-set by Railway (don't manually set this)

## ❓ Troubleshooting:

**Still seeing PORT errors?**
- Check Railway build logs - ensure it's using the Dockerfile
- Verify start.sh has execute permissions (chmod +x)
- Check if Railway is setting the PORT variable

**Container starts but crashes?**
- Check for missing QDRANT_URL or QDRANT_API_KEY
- Review environment variables in Railway dashboard
- Check application logs for Python errors

**Health check fails?**
- Ensure your Qdrant instance is accessible
- Verify API keys are correct
- Check firewall/network settings
