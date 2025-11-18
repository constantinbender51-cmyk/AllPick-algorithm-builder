# Quick Setup Guide

## Project Structure Setup

Create the following directory structure:

```
algo-builder/
├── frontend/
│   ├── src/
│   │   └── App.jsx (from artifact: AlgoBuilder AllPick - Trading Algorithm Builder)
│   ├── public/
│   │   └── index.html
│   ├── package.json (from artifact: package.json - Frontend Dependencies)
│   └── tailwind.config.js
├── backend/
│   ├── main.py (from artifact: main.py - FastAPI Main Application)
│   ├── data_fetcher.py (from artifact: data_fetcher.py - Data Fetching Service)
│   ├── feature_engineer.py (from artifact: feature_engineer.py - Feature Engineering Service)
│   ├── normalizer.py (from artifact: normalizer.py - Data Normalization Service)
│   ├── model_trainer.py (from artifact: model_trainer.py - ML Model Training Service)
│   ├── backtester.py (from artifact: backtester.py - Backtesting Engine)
│   └── requirements.txt (from artifact: requirements.txt - Python Dependencies)
├── railway.json (from artifact: railway.json - Railway Deployment Config)
├── Procfile (from artifact: Procfile - Process Configuration)
├── .gitignore (from artifact: .gitignore - Git Ignore Configuration)
└── README.md (from artifact: README.md - Project Documentation)
```

## Step 1: Create Frontend Structure

### 1.1 Create public/index.html

```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="theme-color" content="#000000" />
    <meta name="description" content="AlgoBuilder AllPick - Trading Algorithm Builder" />
    <title>AlgoBuilder AllPick</title>
  </head>
  <body>
    <noscript>You need to enable JavaScript to run this app.</noscript>
    <div id="root"></div>
  </body>
</html>
```

### 1.2 Create src/index.js

```javascript
import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
```

### 1.3 Create src/index.css

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
    'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
    sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

code {
  font-family: source-code-pro, Menlo, Monaco, Consolas, 'Courier New',
    monospace;
}
```

### 1.4 Create tailwind.config.js

```javascript
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
```

### 1.5 Create postcss.config.js

```javascript
module.exports = {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}
```

## Step 2: Install Dependencies

### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Frontend
```bash
cd frontend
npm install
```

## Step 3: Run Locally

### Terminal 1 - Backend
```bash
cd backend
source venv/bin/activate  # On Windows: venv\Scripts\activate
python main.py
```
Backend will run on http://localhost:8000

### Terminal 2 - Frontend
```bash
cd frontend
npm start
```
Frontend will run on http://localhost:3000

## Step 4: Test the Application

1. Open http://localhost:3000
2. Go through each tab:
   - **Data Selection**: Add a data source (try AAPL from Yahoo Finance)
   - **Features**: Add some technical indicators (SMA, RSI)
   - **Normalization**: Select normalization methods
   - **Target**: Configure prediction target
   - **Algorithm**: Choose an algorithm (Random Forest is fast for testing)
   - **Strategy**: Set strategy parameters
   - **Evaluation**: Click "RUN BACKTEST"

## Step 5: Deploy to Railway

### Method 1: Single Command
```bash
# Initialize git if not already done
git init
git add .
git commit -m "Initial commit"

# Create GitHub repository and push
gh repo create algo-builder --public --source=. --remote=origin --push
```

### Method 2: Railway CLI
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Create new project
railway init

# Deploy
railway up
```

### Method 3: Railway Dashboard
1. Go to https://railway.app
2. Click "New Project"
3. Connect your GitHub repository
4. Railway will auto-detect both services
5. Set environment variables if needed
6. Deploy!

## Troubleshooting

### Backend Issues

**Issue**: ModuleNotFoundError
**Solution**: Make sure you're in the venv and all packages are installed
```bash
pip install -r requirements.txt
```

**Issue**: Data fetch fails
**Solution**: Check internet connection and API availability

### Frontend Issues

**Issue**: Module not found
**Solution**: Delete node_modules and reinstall
```bash
rm -rf node_modules package-lock.json
npm install
```

**Issue**: Tailwind styles not working
**Solution**: Make sure tailwind.config.js and postcss.config.js are properly configured

### CORS Issues

If you see CORS errors, make sure the backend CORS middleware allows your frontend origin.

## Production Considerations

1. **Environment Variables**: Update API_URL in frontend to point to production backend
2. **Rate Limits**: Consider implementing caching for frequently requested data
3. **Error Handling**: Add proper error boundaries in frontend
4. **Authentication**: Add user authentication for production use
5. **Database**: Replace in-memory storage with proper database
6. **Monitoring**: Add logging and monitoring services

## Next Steps

1. Test all features locally
2. Push to GitHub
3. Deploy to Railway
4. Configure custom domain (optional)
5. Set up continuous deployment

## Support

For issues or questions:
- Check the README.md for detailed documentation
- Review API endpoint documentation at http://localhost:8000/docs
- Check Railway logs for deployment issues

Happy Trading! 🚀📈
