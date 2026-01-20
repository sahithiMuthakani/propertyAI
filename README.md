# PropertyAI - Real Estate Valuation Platform

AI-powered real estate price prediction platform with user management, prediction history, analytics dashboard, and modern web interface.

![Platform](https://img.shields.io/badge/Platform-Windows-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![Flask](https://img.shields.io/badge/Flask-3.0.0-lightgrey)
![ML](https://img.shields.io/badge/ML-Scikit--learn-orange)

## 🌟 Features

### Property Price Prediction
- **AI-Powered Predictions**: Stacking ensemble model (XGBoost + LightGBM + GradientBoosting)
- **Model Performance**: R² = 0.7373, MAE = $46,100, RMSE = $88,700
- **Multi-Country Support**: USA, UK, India, Turkey, Bangladesh, and more
- **21 Features**: Advanced feature engineering with original and derived features
- **Price Range Analysis**: Get predicted price with configurable confidence intervals (±10%, ±20%, ±30%)
- **Instant Results**: Get accurate valuations in seconds

### 📊 Analytics Dashboard (NEW)
- **Real-Time Market Insights**: Live statistics from 212K+ property dataset
- **Interactive Charts**: Built with Chart.js for dynamic visualizations
  - Average Price by Country (Bar Chart)
  - Properties by Country (Doughnut Chart)
  - Price Range Distribution
  - Room Distribution
  - Price per SQM by Country
  - Furnishing Status Distribution
- **Global Property Heatmap**: Interactive Leaflet.js map with price-coded markers
- **KPI Cards**: Total properties, average price, countries coverage, area stats
- **AI-Generated Insights**: Key market observations and trends

![Analytics Dashboard](static/screenshots/Analytics%20Dashboard.png)
*Analytics Dashboard with KPI cards, market insights, and interactive charts*

![AI Chatbot](static/screenshots/AI%20Chatbot.png)
*AI-powered real estate assistant chatbot*

### 📝 Blog Section (NEW)
- **Real Estate Insights**: Curated articles on property investment and valuation
- **Modern Card Layout**: Responsive grid with hover effects
- **Full Article Modal**: Read complete articles in a modal view
- **Categories**: Market Analysis, Investment Tips, AI Technology, and more
- **Reading Time**: Estimated reading time for each article

![Blog Page](static/screenshots/Blog%20Page.png)
*Blog section with real estate insights and investment tips*

### 🤖 AI Chatbot (NEW)
- **Real Estate Assistant**: AI-powered chatbot for property queries
- **Gemini Integration**: Powered by Google Gemini API (with fallback responses)
- **Floating Chat Button**: Always-available chat widget
- **Smart Responses**: Answers questions about property valuation, market trends, and more

### 👤 Profile Management (NEW)
- **Profile Settings**: Update user information and preferences
- **Profile Picture Upload**: Upload and manage profile images
- **Password Change**: Secure password update functionality
- **Activity Tracking**: View prediction history and account activity

### User System
- **Role-Based Access**: Separate experiences for regular users and administrators
- **Prediction History**: Track all your valuations with date, location, and price details
- **User Dashboard**: Personal stats including total predictions, average price, and last prediction date
- **Secure Authentication**: Password hashing, session management, and protected routes

![User Dashboard](static/screenshots/Prediction%20and%20user%20Dashboard.png)
*User dashboard with prediction form and history*

### Modern Web Interface
- **2025 Design Standards**: Glassmorphism, gradient orbs, and micro-animations
- **Dark/Light Theme**: Toggle between themes with persistent preference (localStorage)
- **Responsive Layout**: Works perfectly on desktop, tablet, and mobile
- **Smooth Transitions**: Elegant animations and hover effects
- **Floating Orb Effects**: Animated gradient orbs in background

![Landing Page Dark Mode](static/screenshots/Landing%20page%20Dark%20Mode.png)
*Modern landing page with floating orb effects and glassmorphism design*

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Setup

1. **Navigate to the project directory**:
   ```bash
   cd "d:\Real Estate valuation (Vibe Code)"
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### Start the Application

```bash
python app.py
```

The application will be available at: **http://localhost:5000**

### User Workflows

#### For Regular Users
1. Visit http://localhost:5000 (Landing Page)
2. Click "Get Started" to go to login/register
3. Create an account or sign in
4. Access your personal dashboard at `/home`
5. Make property predictions and view your history
6. Explore analytics at `/analytics` and blog at `/blog`

#### For Administrators
1. Login with an admin account
2. Automatically redirected to `/admin`
3. Access user management and system statistics

## 📁 Project Structure

```
Real Estate valuation (Vibe Code)/
├── datasets/                          # CSV datasets folder
│   └── unified_property_data.csv      # Multi-country property data
├── outputs/                           # Data processing outputs
│   └── unified_property_data.csv
├── templates/                         # HTML templates
│   ├── landing.html                   # Public landing page
│   ├── auth.html                      # Login/Register page
│   ├── user_home.html                 # User dashboard
│   ├── analytics.html                 # Analytics dashboard (NEW)
│   ├── blog.html                      # Blog section (NEW)
│   ├── predict.html                   # Standalone prediction page
│   └── index.html                     # Admin dashboard
├── static/                            # Static assets
│   ├── css/style.css
│   ├── js/app.js
│   └── screenshots/                   # README screenshots (NEW)
├── app.py                             # Flask application
├── database.py                        # Database management
├── improved_price_model.py            # ML model (Training & Prediction)
├── FINAL_MODEL_PERFORMANCE.txt        # Model performance report
├── MODEL_README.md                    # Model usage guide
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
└── QUICKSTART.md                      # Quick start guide
```

## 🎨 Pages & Routes

| Route | Access | Description |
|-------|--------|-------------|
| `/` | Public | Landing page with features and CTA |
| `/auth` | Public | Login and registration forms |
| `/home` | Users | User dashboard with predictions |
| `/admin` | Admin | Admin dashboard with user management |
| `/predict` | Public | Standalone prediction page |
| `/analytics` | Users | Market analytics dashboard (NEW) |
| `/blog` | Public | Real estate blog articles (NEW) |

## 📊 API Endpoints

### Authentication
- `POST /api/register` - Register new user
- `POST /api/login` - Login user
- `POST /api/logout` - Logout user
- `GET /api/me` - Get current user

### Predictions
- `POST /api/predict` - Make a property prediction
- `GET /api/predictions` - Get user's prediction history
- `GET /api/countries` - Get available countries
- `GET /api/cities/<country>` - Get cities for a country

### User Management (Admin)
- `GET /api/users` - Get all users
- `GET /api/users/<id>` - Get specific user
- `PUT /api/users/<id>` - Update user
- `DELETE /api/users/<id>` - Delete user

### Profile (NEW)
- `PUT /api/profile` - Update user profile
- `POST /api/profile/picture` - Upload profile picture
- `PUT /api/change-password` - Change password

### Analytics (NEW)
- `GET /api/analytics/stats` - Get analytics statistics and charts data
- `GET /api/analytics/heatmap` - Get heatmap data with coordinates

### Blog (NEW)
- `GET /api/blogs` - Get all blog articles

### AI Chatbot (NEW)
- `POST /api/chat` - Chat with AI assistant

### System
- `GET /api/stats` - Get system statistics
- `GET /api/model/status` - Get ML model status

## 🎨 Design Features

- **Glassmorphism Effects**: Frosted glass cards with backdrop blur
- **Dark/Light Theme**: Toggle with sun/moon icon, persists via localStorage
- **Gradient Designs**: Beautiful linear gradients throughout
- **Animated Orbs**: Floating gradient orbs in background
- **Grid Pattern**: Subtle grid overlay for depth
- **Micro-animations**: Smooth transitions and hover effects
- **Modern Typography**: Clean Inter font family
- **Responsive Charts**: Interactive Chart.js visualizations
- **Interactive Maps**: Leaflet.js heatmap with popups

## 🔒 Security Features

- **Password Hashing**: Werkzeug's secure password hashing
- **Session Management**: Secure Flask sessions (24-hour expiry)
- **Role-Based Access**: Admin routes protected by decorator
- **Input Validation**: Server-side validation for all inputs
- **SQL Injection Prevention**: Parameterized queries

## 🗄️ Database Schema

### Users Table
- `id`, `username`, `email`, `password`, `full_name`
- `role` (user/admin), `is_active`, `created_at`, `updated_at`
- `profile_picture` (NEW)

### Prediction History Table
- `id`, `user_id`, `country`, `city`, `rooms`, `area_sqm`
- `building_age`, `furnishing_status`, `balcony`
- `predicted_price`, `price_low`, `price_high`, `created_at`

## 🤖 Machine Learning Model

### Model Performance
- **Test R² Score**: 0.7373 (explains 73.7% of price variance)
- **Test MAE**: $46,100 (average prediction error)
- **Test RMSE**: $88,700
- **Cross-Validation**: R² = 0.7459 ± 0.0023 (5-fold)

### Architecture
- **Type**: Stacking Ensemble
- **Base Models**: XGBoost, LightGBM, Gradient Boosting
- **Meta-Learner**: Ridge Regression
- **Total Parameters**: 10.1 MB model size

### Features (21 total)
- **Original (7)**: country, city, rooms, area_sqm, balcony, building_age, furnishing_status
- **Engineered (14)**: city/country avg prices, log transformations, polynomial features, interaction terms

### Data Processing
- **Training Data**: 190K+ properties after outlier removal
- **Split**: 70% train, 20% validation, 10% test
- **Data Leakage**: None (proper train/test separation)
- **Scaling**: RobustScaler (robust to outliers)

For complete model documentation, see `FINAL_MODEL_PERFORMANCE.txt`

## 🆕 What's New

### Version 2.0 Features
1. **Analytics Dashboard** - Interactive market insights with Chart.js and Leaflet.js
2. **Blog Section** - Real estate articles and investment tips
3. **AI Chatbot** - Gemini-powered assistant for property queries
4. **Profile Management** - User profile updates and picture uploads
5. **Enhanced UI** - Refined glassmorphism design across all pages
6. **Dark/Light Mode** - Consistent theme toggle across all pages

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| Port 5000 in use | Change port in `app.py` or kill process |
| Model not loading | Ensure `models/` folder has trained model files |
| Database locked | Close other connections to SQLite file |
| Charts not loading | Check browser console for JavaScript errors |
| Heatmap not showing | Ensure internet connection for Leaflet tiles |

## 📝 License

This project is provided as-is for educational and development purposes.

## 👨‍💻 Author

Created with ❤️ using Python, Flask, and Modern Web Technologies - 2025
