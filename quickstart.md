# Quick Start Guide

## 🚀 Getting Started in 2 Minutes

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Launch the Application

```bash
python app.py
```

### Step 3: Open Your Browser

Navigate to: **http://localhost:5000**

---

## 🎯 What You'll See

### 1. Landing Page (`/`)
- Beautiful hero section with animated background
- Feature showcase with 6 cards
- "How It Works" section
- Dark/Light theme toggle (sun/moon icon in nav)

### 2. Create an Account (`/auth`)
- Click "Get Started" or "Login" 
- Switch to "Create Account" tab
- Fill in: username, email, password
- Click "Create Account"

### 3. User Dashboard (`/home`)
After registration, you'll see:
- **Stats Cards**: Total predictions, average price, last prediction
- **Prediction Form**: Select country, city, rooms, area, etc.
- **History Table**: All your past predictions

### 4. Make Your First Prediction
1. Select a **Country** (e.g., USA)
2. Select a **City** (e.g., New York)
3. Enter **Rooms** and **Area (sqm)**
4. Click **"Get Price Prediction"**
5. See the estimated price with range!

---

## 🌙 Try the Theme Toggle

- Look for the **moon icon** in the top navigation
- Click it to switch to **light mode**
- Click the **sun icon** to switch back to dark mode
- Your preference is saved automatically!

---

## 👤 User Roles

| Role | After Login | Access |
|------|-------------|--------|
| **User** | → `/home` | Dashboard, Predictions, History |
| **Admin** | → `/admin` | User Management, System Stats |

*New accounts are created as regular users by default.*

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| `app.py` | Main Flask application |
| `database.py` | SQLite database management |
| `price_prediction.py` | ML model for predictions |
| `templates/landing.html` | Public landing page |
| `templates/auth.html` | Login/Register page |
| `templates/user_home.html` | User dashboard |

---

## 💡 Pro Tips

- **Prediction History**: All your predictions are saved automatically
- **Theme Preference**: Persists across all pages and sessions
- **Price Range**: Choose ±5%, ±10%, or ±15% confidence
- **Logout**: Click the red "Logout" button in the header

---

## 🎨 Enjoy the Experience!

The interface features:
- ✨ Glassmorphism effects
- 🌈 Smooth gradient animations
- 🌙 Dark/Light mode toggle
- 📱 Responsive design
- ⚡ Instant predictions

**Start exploring your property valuations!** 🏠
