# 📱 VERCEL + RAILWAY DEPLOYMENT - COMPLETE GUIDE

> Everything you need to deploy your full-stack app in 15 minutes!

---

## 🎯 DEPLOYMENT ROADMAP

```
┌─────────────────────────────┐
│ Your GitHub Repository      │
│ SynFeasNet                  │
└─────────────────────────────┘
         ↓ (Push to main)
┌─────────────────────────────┐
│ VERCEL (React Frontend)     │
│ synfeasnet.vercel.app       │
└─────────────────────────────┘
         ↓ (API Calls)
┌─────────────────────────────┐
│ RAILWAY (Python Backend)    │
│ synfeasnet-prod.railway.app │
└─────────────────────────────┘
```

---

## ⚡ SUPER QUICK START (15 MINUTES TOTAL)

### Frontend Deployment (5 min)
```bash
1. Open: https://vercel.com/signup
2. Click: "Continue with GitHub"
3. Select: "SynFeasNet" repo
4. Set Root Directory: "./frontend"
5. Click: "Deploy"
✅ Live at: https://synfeasnet.vercel.app
```

### Backend Deployment (10 min)
```bash
1. Open: https://railway.app/signup
2. Click: "Continue with GitHub"
3. Click: "New Project" → "Deploy from GitHub"
4. Select: "SynFeasNet" repo
5. Wait for deployment
6. Copy: Backend URL from Settings
✅ Live at: https://synfeasnet-prod-xxx.railway.app
```

### Connect Them (1 min)
```bash
1. Vercel Dashboard → Settings → Environment Variables
2. Add: VITE_API_URL = https://synfeasnet-prod-xxx.railway.app
3. Save & Redeploy
✅ Connected!
```

---

## 📋 DETAILED SETUP

### PART 1: VERCEL SETUP (React Frontend)

#### Step 1.1: Create Vercel Account
- Go: https://vercel.com
- Click: "Sign Up"
- Click: "Continue with GitHub"
- Authorize Vercel

#### Step 1.2: Import Project
- Click: "Add New..." → "Project"
- Search: "SynFeasNet"
- Click: "Import"

#### Step 1.3: Configure
```
Project Name: SynFeasNet
Framework: Vite (auto-detected)
Root Directory: ./frontend ⚠️ IMPORTANT
Build Command: npm run build
Output Directory: dist
```

**Must set Root Directory to `./frontend`!**

#### Step 1.4: Deploy
- Click: "Deploy" (blue button)
- Wait 2-3 minutes
- ✅ Your app is live!

**Your Frontend URL:**
```
https://synfeasnet.vercel.app
```

---

### PART 2: RAILWAY SETUP (Python Backend)

#### Step 2.1: Create Railway Account
- Go: https://railway.app
- Click: "Login"
- Click: "Continue with GitHub"
- Authorize Railway

#### Step 2.2: Create Project
- Click: "New Project"
- Click: "Deploy from GitHub repo"
- Search: "SynFeasNet"
- Select: "Rahul-07-code/SynFeasNet"

#### Step 2.3: Configure (Optional)
- Click: "Variables" tab
- Add:
  ```
  PYTHON_VERSION = 3.11
  AWS_REGION = us-east-1
  ```

#### Step 2.4: Deploy
- Click: "Deploy"
- Watch logs scroll
- Wait 5-10 minutes
- ✅ Your backend is live!

#### Step 2.5: Get Backend URL
- Click: "Settings" tab
- Scroll to: "Networking" section
- Copy: "Domain" URL
  
**Your Backend URL:**
```
https://synfeasnet-prod-abc123.railway.app
```

---

### PART 3: CONNECT FRONTEND TO BACKEND (1 minute)

#### Step 3.1: Set Environment Variable
Go to **Vercel Dashboard**:

1. Click: "SynFeasNet" project
2. Click: "Settings"
3. Click: "Environment Variables"
4. Click: "Add New Variable"
5. Fill in:
   ```
   Name: VITE_API_URL
   Value: https://synfeasnet-prod-abc123.railway.app
   ```
6. Click: "Save"

**Vercel automatically redeploys!** ✅

#### Step 3.2: Verify Connection
1. Visit: https://synfeasnet.vercel.app
2. Enter SMILES: "CCO" (ethanol)
3. Click: "Predict"
4. Check if results appear ✅

---

## 🔄 AUTOMATIC DEPLOYMENTS

After setup, you get automatic deployments:

```bash
# You push to GitHub
git add .
git commit -m "Update feature"
git push origin main

# Vercel auto-redeploys in 1-2 minutes ✅
# Railway auto-redeploys in 5-10 minutes ✅
```

No more manual deployments needed! 🎉

---

## 📊 MONITORING

### Check Vercel Status
1. Dashboard → Deployments
2. See real-time deployment status
3. View build logs

### Check Railway Status
1. Dashboard → Logs
2. See real-time application logs
3. Monitor for errors

---

## 🆘 COMMON ISSUES & FIXES

### Issue: "Build failed on Vercel"
```
Solution:
1. Check build logs in Deployments
2. Common fixes:
   - npm ci (install dependencies)
   - Check Node version (use 18+)
   - Verify package.json exists in frontend/
```

### Issue: "Frontend can't call backend API"
```
Solution:
1. Verify VITE_API_URL is set in Vercel
2. Check backend is running on Railway
3. Enable CORS in backend (if needed)
4. Test URL in browser console:
   console.log(import.meta.env.VITE_API_URL)
```

### Issue: "Railway build failed"
```
Solution:
1. Check Railway build logs
2. Common fixes:
   - pip install -r requirements.txt
   - Check Python version (3.11+)
   - Verify model file exists
```

### Issue: "Timeout errors from Railway"
```
Solution:
1. Railway free tier might be slow
2. Upgrade to paid plan ($5+/month)
3. Or optimize your Python code
```

---

## 💰 COSTS

| Service | Free Tier | Pricing |
|---------|-----------|---------|
| **Vercel** | Unlimited | $20/month Pro (optional) |
| **Railway** | $5 credit/month | $0.12/CPU-hour after free tier |
| **Total** | ✅ Completely FREE | ~$0-5/month after free tier |

---

## 🎓 WHAT YOU'VE DONE

✅ Deployed React frontend on Vercel  
✅ Deployed Python backend on Railway  
✅ Connected frontend to backend  
✅ Set up automatic deployments from GitHub  
✅ Created a full-stack production application  

---

## 📚 NEXT STEPS

1. **Custom Domain** (optional)
   - Buy domain (Namecheap, GoDaddy)
   - Connect to Vercel
   - Instructions: https://vercel.com/docs/custom-domains

2. **SSL Certificate** (automatic)
   - Already enabled on both services
   - No configuration needed

3. **Analytics** (optional)
   - Vercel: Dashboard → Analytics
   - Railway: Dashboard → Metrics

4. **Error Monitoring** (optional)
   - Sentry integration
   - Better error tracking

---

## 📞 SUPPORT

**If something breaks:**

1. Check the logs
   - Vercel: Dashboard → Deployments → Logs
   - Railway: Dashboard → Logs

2. Common fixes:
   - Redeploy: Push to GitHub again
   - Clear cache: In Vercel settings
   - Check environment variables

3. Get help:
   - Vercel Docs: https://vercel.com/docs
   - Railway Docs: https://docs.railway.app
   - Create GitHub issue

---

## 🎉 CONGRATULATIONS!

Your app is now:
- ✅ **Live** on the internet
- ✅ **Accessible** from any browser
- ✅ **Auto-deploying** from GitHub
- ✅ **Scalable** with zero configuration
- ✅ **Free** (or nearly free)

**Share your app URL:**
```
https://synfeasnet.vercel.app
```

🚀 **You're a deployed developer!** 🚀

---

## 🔗 USEFUL LINKS

- Vercel: https://vercel.com
- Railway: https://railway.app
- GitHub: https://github.com/Rahul-07-code/SynFeasNet
- Your App: https://synfeasnet.vercel.app
- Your API: https://synfeasnet-prod-xxx.railway.app

---

**Questions? Create an issue on GitHub!** 📝
