# Frontend Files Summary

## Complete File Listing

Here's every file you need for the frontend:

### 📁 Root Level Files

1. **package.json**
   - Location: `frontend/package.json`
   - Artifact: "package.json - Frontend Dependencies"
   - Contains: All npm dependencies (React, Recharts, Lucide icons, Tailwind)

2. **tailwind.config.js**
   - Location: `frontend/tailwind.config.js`
   - Artifact: "tailwind.config.js"
   - Contains: Tailwind CSS configuration, custom colors, animations

3. **postcss.config.js**
   - Location: `frontend/postcss.config.js`
   - Artifact: "postcss.config.js"
   - Contains: PostCSS configuration for Tailwind processing

4. **.env.example**
   - Location: `frontend/.env.example`
   - Artifact: ".env.example (Frontend)"
   - Contains: Environment variable template

### 📁 public/ Directory

5. **index.html**
   - Location: `frontend/public/index.html`
   - Artifact: "public/index.html"
   - Contains: HTML template with meta tags and root div

### 📁 src/ Directory

6. **index.js**
   - Location: `frontend/src/index.js`
   - Artifact: "src/index.js"
   - Contains: React app entry point, renders App component

7. **index.css**
   - Location: `frontend/src/index.css`
   - Artifact: "src/index.css"
   - Contains: Tailwind imports, global styles, custom scrollbar, animations

8. **App.jsx**
   - Location: `frontend/src/App.jsx`
   - Artifact: "AlgoBuilder AllPick - Trading Algorithm Builder"
   - Contains: **ENTIRE APPLICATION** - all 7 tabs, state management, UI logic

## File Dependencies

```
index.html (root template)
    ↓
index.js (React entry)
    ↓
    ├── index.css (Tailwind + global styles)
    └── App.jsx (main application)
```

## Installation Steps

### Step 1: Create Directory Structure
```bash
mkdir -p frontend/public frontend/src
cd frontend
```

### Step 2: Copy Files

Copy each artifact to its location:

```bash
# Root level
touch package.json tailwind.config.js postcss.config.js .env.example

# Public directory
touch public/index.html

# Src directory
touch src/index.js src/index.css src/App.jsx
```

### Step 3: Install Dependencies
```bash
npm install
```

This will install:
- **react** (^18.2.0) - Core React library
- **react-dom** (^18.2.0) - React DOM rendering
- **react-scripts** (5.0.1) - Create React App scripts
- **recharts** (^2.10.3) - Charting library for equity curves
- **lucide-react** (^0.294.0) - Icon library
- **axios** (^1.6.2) - HTTP client for API calls
- **tailwindcss** (^3.3.0) - Utility-first CSS framework
- **autoprefixer** (^10.4.16) - CSS vendor prefixing
- **postcss** (^8.4.32) - CSS processing

### Step 4: Configure Environment
```bash
cp .env.example .env.local
# Edit .env.local if needed (default is correct for local dev)
```

### Step 5: Start Development Server
```bash
npm start
```

Application will open at `http://localhost:3000`

## File Contents Quick Reference

### App.jsx (Main Application)
This is the largest file and contains:

**State Management:**
- `dataSources` - Added data sources
- `features` - Technical indicators
- `normalization` - Normalization config per feature
- `target` - Prediction target config
- `algorithm` - Selected algorithm and hyperparameters
- `strategy` - Trading strategy parameters
- `backtestResults` - Results from backtest

**7 Tab Components:**
1. `renderDataTab()` - Data source selection
2. `renderFeaturesTab()` - Feature engineering
3. `renderNormalizationTab()` - Normalization configuration
4. `renderTargetTab()` - Target definition
5. `renderAlgorithmTab()` - Algorithm selection
6. `renderStrategyTab()` - Strategy configuration
7. `renderEvaluationTab()` - Results visualization

**Key Functions:**
- `addDataSource()` - Fetch data from API
- `addFeature()` - Add technical indicator
- `updateNormalization()` - Configure normalization
- `runBacktest()` - Execute full pipeline
- `previewNormalization()` - Preview normalization effects

## Important Notes

### API_URL Configuration
The frontend expects the backend at `http://localhost:8000/api` by default.

To change this:
1. Edit the `API_URL` constant in `App.jsx` (line 4)
2. Or use environment variable: `REACT_APP_API_URL`

### For Production Deployment
Update `API_URL` to your Railway backend URL:
```javascript
const API_URL = process.env.REACT_APP_API_URL || 'https://your-backend.railway.app/api';
```

### Mobile Optimization
The app automatically detects screen size and switches between:
- **Desktop**: Horizontal tabs
- **Mobile**: Vertical stepper

Breakpoint is at 768px (md: in Tailwind)

## Troubleshooting

### Issue: "Module not found: Can't resolve 'react'"
**Solution**: Install dependencies
```bash
npm install
```

### Issue: Tailwind styles not appearing
**Solution**: 
1. Check `tailwind.config.js` exists
2. Check `postcss.config.js` exists
3. Restart dev server: `npm start`

### Issue: CORS errors
**Solution**: Make sure backend is running and CORS is configured correctly in `backend/main.py`

### Issue: Charts not displaying
**Solution**: Check that recharts is installed:
```bash
npm install recharts
```

## Build for Production

```bash
npm run build
```

This creates an optimized production build in the `build/` directory.

## File Size Summary

Approximate file sizes:
- `App.jsx`: ~25 KB (the entire application!)
- `package.json`: ~1 KB
- `index.js`: ~300 bytes
- `index.css`: ~1 KB
- `index.html`: ~1 KB
- `tailwind.config.js`: ~800 bytes
- `postcss.config.js`: ~100 bytes

**Total source code**: ~28 KB (incredibly compact!)

## Next Steps

1. ✅ Create all files as listed above
2. ✅ Run `npm install`
3. ✅ Start dev server with `npm start`
4. ✅ Test the application
5. ✅ Build for production with `npm run build`
6. ✅ Deploy to Railway

The entire frontend is essentially one large React component (`App.jsx`) that manages all the state and renders different tabs based on user interaction. This makes it very easy to maintain and deploy!
