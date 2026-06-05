# 🚀 VERCEL DEPLOYMENT GUIDE - SynFeasNet React Frontend

> **Status:** Complete step-by-step guide for deploying your React app on Vercel

---

## ⚡ WHAT IS VERCEL?

- **Easy**: One-click deployment from GitHub
- **Free**: No credit card needed for free tier
- **Fast**: Automatic CDN and optimizations
- **Perfect for**: React/Next.js/Node.js apps

---

## 📋 STEP-BY-STEP DEPLOYMENT

### STEP 1: Create Vercel Account (2 minutes)

1. Go to: https://vercel.com
2. Click **"Sign Up"** (top right)
3. Click **"Continue with GitHub"**
4. Authorize Vercel to access your GitHub
5. You'll be logged in automatically ✅

### STEP 2: Connect Your Repository (2 minutes)

1. After login, you'll see Vercel Dashboard
2. Click **"Add New..."** → **"Project"**
3. Click **"Import Git Repository"**
4. Search for: **"SynFeasNet"**
5. Click on **"Rahul-07-code/SynFeasNet"**
6. Click **"Import"**

### STEP 3: Configure Project Settings (3 minutes)

You'll see this screen:

```
┌─────────────────────────────────┐
│ Configure Project               │
│                                 │
│ Project Name: SynFeasNet        │
│ Framework: Vite                 │
│ Root Directory: ./frontend      │
│                                 │
│ Build Command: npm run build    │
│ Output Directory: dist          │
└─────────────────────────────────┘
```

**Settings to check:**

| Setting | Value | Action |
|---------|-------|--------|
| Project Name | SynFeasNet | ✅ Keep it |
| Framework | Vite | ✅ Keep it |
| Root Directory | `./frontend` | ✅ **IMPORTANT: Set this** |
| Build Command | `npm run build` | ✅ Keep it |
| Output Directory | `dist` | ✅ Keep it |
| Install Command | `npm ci` | ✅ Keep default |

**Most important:** Set **Root Directory to `./frontend`** because your React app is in the `frontend/` folder.

### STEP 4: Deploy! (5-10 minutes)

1. Click **"Deploy"** button (blue)
2. Wait for deployment to complete
3. You'll see: **"Congratulations! Your project has been successfully deployed"**
4. Your app URL: `https://synfeasnet.vercel.app` ✅

---

## ✅ YOUR APP IS NOW LIVE!

Visit: **https://synfeasnet.vercel.app**

---

## 🔗 CONNECTING BACKEND (Python API)

Your React frontend needs to communicate with your Python backend. Here's how:

### Option 1: Use External Python API (Recommended)

If you deploy Python backend separately (AWS, Railway, Heroku, etc.):

**Create `.env.local` in your `frontend/` folder:**

```env
VITE_API_URL=https://your-python-backend.com
```

**Use in React:**
```javascript
const API_URL = import.meta.env.VITE_API_URL;

async function predict(smiles) {
  const response = await fetch(`${API_URL}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ smiles })
  });
  return response.json();
}
```

### Option 2: Use Vercel API Routes (Serverless)

Create `frontend/api/predict.ts`:

```typescript
import { VercelRequest, VercelResponse } from '@vercel/node';

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method === 'POST') {
    const { smiles } = req.body;
    
    // Call your Python backend here
    const response = await fetch('https://your-python-api.com/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ smiles })
    });
    
    const data = await response.json();
    res.status(200).json(data);
  }
}
```

Then call from React:
```javascript
async function predict(smiles) {
  const response = await fetch('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ smiles })
  });
  return response.json();
}
```

---

## 🔄 AUTOMATIC DEPLOYMENTS

Every time you push to GitHub:

```bash
git add .
git commit -m "Update frontend"
git push origin main
```

✅ **Vercel automatically redeploys!** (Takes 1-2 minutes)

Check deployment status: https://vercel.com/dashboard

---

## 🌐 CUSTOM DOMAIN (Optional)

Want: `synfeasnet.com` instead of `synfeasnet.vercel.app`?

1. Go to Vercel Dashboard
2. Select your project
3. Click **"Settings"** → **"Domains"**
4. Click **"Add Domain"**
5. Enter your domain
6. Follow DNS setup instructions

---

## 📊 MONITORING & LOGS

### View Deployment Logs

1. Go to Vercel Dashboard
2. Click on **"SynFeasNet"** project
3. Click **"Deployments"** tab
4. Click any deployment
5. Scroll to **"Logs"**

### View Application Errors

1. Click **"Functions"** tab
2. Click on any error to debug

---

## 🔧 ENVIRONMENT VARIABLES

If your frontend needs secrets (API keys, etc.):

1. Go to project **Settings** → **Environment Variables**
2. Add variables:
   ```
   VITE_API_URL = https://your-api.com
   VITE_API_KEY = your_secret_key
   ```
3. Redeploy project
4. Access in code:
   ```javascript
   const apiUrl = import.meta.env.VITE_API_URL;
   ```

---

## 🚀 DEPLOY PYTHON BACKEND

You have multiple options:

### Option A: Railway (Recommended - Free tier available)
```bash
# 1. Go to: https://railway.app
# 2. Connect GitHub
# 3. Select SynFeasNet repo
# 4. Create service for Python
# 5. Deploy automatically
```

**Example Procfile:**
```
web: streamlit run app/app.py --server.port=$PORT
```

### Option B: Render.com (Also free)
```bash
# 1. Go to: https://render.com
# 2. Connect GitHub
# 3. Create Web Service
# 4. Select Python
# 5. Deploy
```

### Option C: AWS (Covered earlier)
```bash
# Using Elastic Beanstalk, Lambda, or ECS
```

---

## 📝 VERCEL DEPLOYMENT CHECKLIST

- [ ] Created Vercel account
- [ ] Connected GitHub repository
- [ ] Set Root Directory to `./frontend`
- [ ] Clicked Deploy
- [ ] App is live at https://synfeasnet.vercel.app
- [ ] Python backend URL configured
- [ ] Environment variables set (if needed)
- [ ] Custom domain added (optional)

---

## 🆘 TROUBLESHOOTING

### Issue: "Build failed"

**Solution:** Check build logs
```
1. Go to Vercel Dashboard
2. Click "Deployments"
3. Find failed deployment
4. View "Build Logs"
5. Look for error message
```

Common fix:
```bash
# Ensure you have package.json in frontend folder
ls frontend/package.json

# And vercel.json configured correctly
cat frontend/vercel.json
```

### Issue: "Module not found"

**Solution:** Install dependencies
```bash
cd frontend
npm install
npm run build
```

### Issue: "Port already in use"

This doesn't happen on Vercel (serverless). If testing locally:
```bash
# Kill process on port 5173
lsof -ti:5173 | xargs kill -9

# Or use different port
npm run dev -- --port 3000
```

### Issue: "CORS errors"

**Solution:** Use Vercel API Routes (Option 2 above) or configure backend CORS:

**In Python (FastAPI/Flask):**
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://synfeasnet.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Issue: "Frontend works but API calls fail"

**Solution:** Check if API URL is correct
```javascript
// In browser console, check:
console.log(import.meta.env.VITE_API_URL);

// Should show your backend URL, not undefined
```

---

## 🎯 NEXT STEPS

1. **Deploy Python Backend**
   - Choose: Railway, Render, or AWS
   - Get backend URL

2. **Connect Frontend to Backend**
   - Add `VITE_API_URL` environment variable
   - Update React API calls

3. **Test End-to-End**
   - Submit SMILES string
   - Verify prediction works
   - Check results display

4. **Monitor Performance**
   - Go to Vercel Analytics
   - Check page load times
   - Optimize if needed

---

## 💡 PRO TIPS

✅ **Cache Static Assets**
- Vercel handles this automatically
- Your CSS/JS files are cached globally

✅ **Enable Analytics**
- Vercel Dashboard → Analytics
- See real user traffic

✅ **Preview Deployments**
- Pull requests automatically get preview URLs
- Test before merging to main

✅ **Automatic HTTPS**
- All Vercel apps have SSL by default
- No configuration needed

---

## 📚 RESOURCES

- Vercel Docs: https://vercel.com/docs
- React with Vercel: https://vercel.com/frameworks/next-js
- Environment Variables: https://vercel.com/docs/projects/environment-variables
- Custom Domains: https://vercel.com/docs/custom-domains

---

## ✨ SUMMARY

```
GitHub Push
    ↓
Vercel Detects Change
    ↓
Automatic Build & Deploy
    ↓
Live at https://synfeasnet.vercel.app ✅
```

**That's it! Your React app is live and updates automatically!** 🎉

---

## 🎬 QUICK START VIDEO SUMMARY

1. Create account: https://vercel.com/signup
2. Import project from GitHub
3. Set root directory to `./frontend`
4. Click Deploy
5. Done! 🚀

---

Need help? Create a GitHub issue with your error message!
