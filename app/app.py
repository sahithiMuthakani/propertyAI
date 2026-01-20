"""
Flask User Management and Property Price Prediction Application
================================================================

A modern user management system with REST API endpoints and property price prediction.
Features: User registration, authentication, profile management, admin dashboard,
and multi-country property price prediction.

Date: 2025-12-04
Updated: 2025-12-08
"""

from flask import Flask, request, jsonify, session, render_template, send_from_directory, redirect
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from database import Database
import os
import json
from datetime import datetime, timedelta
import secrets
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Gemini AI Integration (using google-genai SDK)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
try:
    from google import genai
    genai_client = genai.Client(api_key=GEMINI_API_KEY)
    CHATBOT_ENABLED = True
    print("✓ Gemini AI chatbot initialized")
except Exception as e:
    CHATBOT_ENABLED = False
    genai_client = None
    print(f"⚠ Gemini chatbot not available: {e}")

# Import price prediction model
try:
    from improved_price_model import ImprovedPricePredictionModel
    price_model = ImprovedPricePredictionModel()
    price_model.load_model()
    PREDICTION_ENABLED = True
    print("✓ Improved price prediction model loaded successfully")
except Exception as e:
    PREDICTION_ENABLED = False
    price_model = None
    print(f"⚠ Price prediction model not loaded: {e}")

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)  # Secure secret key
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)

# Enable CORS
CORS(app)

# Initialize database
db = Database()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def user_to_dict(user, include_password=False):
    """Convert user dict to safe format for API response"""
    if not user:
        return None
    
    user_dict = {
        'id': user['id'],
        'username': user['username'],
        'email': user['email'],
        'full_name': user.get('full_name'),
        'role': user['role'],
        'profile_picture': user.get('profile_picture'),
        'phone': user.get('phone'),
        'location': user.get('location'),
        'user_city': user.get('user_city'),
        'user_country': user.get('user_country'),
        'created_at': user['created_at'],
        'updated_at': user.get('updated_at'),
        'is_active': user['is_active']
    }
    
    if include_password:
        user_dict['password'] = user['password']
    
    return user_dict


def require_login(f):
    """Decorator to require user login"""
    def wrapper(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Authentication required'}), 401
        return f(*args, **kwargs)
    wrapper.__name__ = f.__name__
    return wrapper


def require_admin(f):
    """Decorator to require admin role"""
    def wrapper(*args, **kwargs):
        if 'user_id' not in session:
            return redirect('/auth')
        user = db.get_user_by_id(session['user_id'])
        if not user or user['role'] != 'admin':
            return redirect('/home')
        return f(*args, **kwargs)
    wrapper.__name__ = f.__name__
    return wrapper


# ============================================================================
# FRONTEND ROUTES
# ============================================================================

@app.route('/')
def landing():
    """Serve public landing page"""
    return render_template('landing.html')


@app.route('/auth')
def auth_page():
    """Serve authentication page (login/register)"""
    # If already logged in, redirect to appropriate page
    if 'user_id' in session:
        user = db.get_user_by_id(session['user_id'])
        if user and user['role'] == 'admin':
            return redirect('/admin')
        return redirect('/home')
    return render_template('auth.html')


@app.route('/home')
def user_home():
    """Serve user dashboard - requires login"""
    if 'user_id' not in session:
        return redirect('/auth')
    return render_template('user_home.html')


@app.route('/admin')
@require_admin
def admin_dashboard():
    """Serve admin dashboard - requires admin role"""
    return render_template('index.html')


@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory('static', path)


# ============================================================================
# API ENDPOINTS - Authentication
# ============================================================================

@app.route('/api/register', methods=['POST'])
def register():
    """Register a new user"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['username', 'email', 'password']
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({'error': f'{field} is required'}), 400
        
        username = data['username'].strip()
        email = data['email'].strip().lower()
        password = data['password']
        full_name = data.get('full_name', '').strip()
        
        # Validate email format
        if '@' not in email:
            return jsonify({'error': 'Invalid email format'}), 400
        
        # Check if user already exists
        if db.get_user_by_email(email):
            return jsonify({'error': 'Email already registered'}), 409
        
        if db.get_user_by_username(username):
            return jsonify({'error': 'Username already taken'}), 409
        
        # Hash password
        hashed_password = generate_password_hash(password)
        
        # Create user
        user_id = db.create_user(
            username=username,
            email=email,
            password=hashed_password,
            full_name=full_name if full_name else None
        )
        
        if user_id:
            # Get created user
            user = db.get_user_by_id(user_id)
            
            # Create session
            session['user_id'] = user_id
            session['username'] = username
            session.permanent = True
            
            return jsonify({
                'message': 'User registered successfully',
                'user': user_to_dict(user)
            }), 201
        else:
            return jsonify({'error': 'Failed to create user'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/login', methods=['POST'])
def login():
    """Login user"""
    try:
        data = request.get_json()
        
        # Validate required fields
        if 'email' not in data or 'password' not in data:
            return jsonify({'error': 'Email and password are required'}), 400
        
        email = data['email'].strip().lower()
        password = data['password']
        
        # Get user
        user = db.get_user_by_email(email)
        
        if not user:
            return jsonify({'error': 'Invalid email or password'}), 401
        
        # Check if user is active
        if not user['is_active']:
            return jsonify({'error': 'Account is inactive'}), 403
        
        # Verify password
        if not check_password_hash(user['password'], password):
            return jsonify({'error': 'Invalid email or password'}), 401
        
        # Create session
        session['user_id'] = user['id']
        session['username'] = user['username']
        session.permanent = True
        
        return jsonify({
            'message': 'Login successful',
            'user': user_to_dict(user)
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/logout', methods=['POST'])
@require_login
def logout():
    """Logout user"""
    session.clear()
    return jsonify({'message': 'Logout successful'}), 200


@app.route('/api/me', methods=['GET'])
@require_login
def get_current_user():
    """Get current logged in user"""
    user_id = session.get('user_id')
    user = db.get_user_by_id(user_id)
    
    if user:
        return jsonify({'user': user_to_dict(user)}), 200
    else:
        return jsonify({'error': 'User not found'}), 404


# ============================================================================
# API ENDPOINTS - User Management
# ============================================================================

@app.route('/api/users', methods=['GET'])
@require_login
def get_users():
    """Get all users"""
    try:
        users = db.get_all_users()
        users_list = [user_to_dict(user) for user in users]
        
        return jsonify({
            'users': users_list,
            'total': len(users_list)
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/users/<int:user_id>', methods=['GET'])
@require_login
def get_user(user_id):
    """Get specific user by ID"""
    try:
        user = db.get_user_by_id(user_id)
        
        if user:
            return jsonify({'user': user_to_dict(user)}), 200
        else:
            return jsonify({'error': 'User not found'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/users/<int:user_id>', methods=['PUT'])
@require_login
def update_user(user_id):
    """Update user information"""
    try:
        # Check if updating own profile or admin
        current_user_id = session.get('user_id')
        current_user = db.get_user_by_id(current_user_id)
        
        if current_user_id != user_id and current_user['role'] != 'admin':
            return jsonify({'error': 'Unauthorized'}), 403
        
        data = request.get_json()
        
        # Prepare update data
        update_data = {}
        
        if 'username' in data:
            update_data['username'] = data['username'].strip()
        
        if 'email' in data:
            update_data['email'] = data['email'].strip().lower()
        
        if 'full_name' in data:
            update_data['full_name'] = data['full_name'].strip()
        
        if 'password' in data and data['password']:
            update_data['password'] = generate_password_hash(data['password'])
        
        # Only admin can change role
        if 'role' in data and current_user['role'] == 'admin':
            update_data['role'] = data['role']
        
        # Update user
        if db.update_user(user_id, **update_data):
            updated_user = db.get_user_by_id(user_id)
            return jsonify({
                'message': 'User updated successfully',
                'user': user_to_dict(updated_user)
            }), 200
        else:
            return jsonify({'error': 'Failed to update user'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/users/<int:user_id>', methods=['DELETE'])
@require_login
def delete_user(user_id):
    """Delete user (soft delete)"""
    try:
        # Check if admin
        current_user_id = session.get('user_id')
        current_user = db.get_user_by_id(current_user_id)
        
        if current_user['role'] != 'admin' and current_user_id != user_id:
            return jsonify({'error': 'Unauthorized'}), 403
        
        # Prevent deleting self
        if current_user_id == user_id:
            return jsonify({'error': 'Cannot delete your own account'}), 400
        
        # Soft delete user
        if db.delete_user(user_id, soft_delete=True):
            return jsonify({'message': 'User deleted successfully'}), 200
        else:
            return jsonify({'error': 'Failed to delete user'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# API ENDPOINTS - Statistics
# ============================================================================

@app.route('/api/stats', methods=['GET'])
@require_login
def get_stats():
    """Get system statistics"""
    try:
        total_users = db.count_users(active_only=False)
        active_users = db.count_users(active_only=True)
        
        return jsonify({
            'total_users': total_users,
            'active_users': active_users,
            'inactive_users': total_users - active_users
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# API ENDPOINTS - Price Prediction
# ============================================================================

@app.route('/predict')
def predict_page():
    """Serve the price prediction page"""
    return render_template('predict.html')


@app.route('/api/countries', methods=['GET'])
def get_countries():
    """Get list of available countries for prediction"""
    if not PREDICTION_ENABLED:
        return jsonify({'error': 'Prediction service not available'}), 503
    
    try:
        dropdown_data = price_model.get_dropdown_data()
        return jsonify({'countries': dropdown_data['countries']}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/cities/<country>', methods=['GET'])
def get_cities(country):
    """Get list of cities for a specific country"""
    if not PREDICTION_ENABLED:
        return jsonify({'error': 'Prediction service not available'}), 503
    
    try:
        dropdown_data = price_model.get_dropdown_data()
        cities = dropdown_data['cities_by_country'].get(country, [])
        return jsonify({'cities': cities}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict_price():
    """Predict property price based on input features"""
    if not PREDICTION_ENABLED:
        return jsonify({'error': 'Prediction service not available'}), 503
    
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['country', 'city', 'rooms', 'area_sqm']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'{field} is required'}), 400
        
        # Prepare input data
        input_data = {
            'country': data['country'],
            'city': data['city'],
            'rooms': int(data['rooms']),
            'area_sqm': float(data['area_sqm']),
            'balcony': data.get('balcony', False),
            'building_age': int(data.get('building_age', 10)),
            'furnishing_status': data.get('furnishing_status', 'Unknown')
        }
        
        # Get prediction with range
        range_pct = float(data.get('range_pct', 10))
        result = price_model.predict(input_data, range_pct=range_pct)
        
        # Save prediction for logged-in users
        prediction_id = None
        if 'user_id' in session:
            prediction_id = db.save_prediction(
                user_id=session['user_id'],
                country=input_data['country'],
                city=input_data['city'],
                rooms=input_data['rooms'],
                area_sqm=input_data['area_sqm'],
                building_age=input_data['building_age'],
                furnishing_status=input_data['furnishing_status'],
                balcony=input_data['balcony'],
                predicted_price=result['predicted_price'],
                price_low=result['low'],
                price_high=result['high']
            )
        
        return jsonify({
            'success': True,
            'prediction': result,
            'prediction_id': prediction_id,
            'input': input_data
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predictions', methods=['GET'])
@require_login
def get_predictions():
    """Get current user's prediction history"""
    try:
        user_id = session.get('user_id')
        limit = request.args.get('limit', 50, type=int)
        
        predictions = db.get_user_predictions(user_id, limit=limit)
        stats = db.get_user_prediction_stats(user_id)
        
        return jsonify({
            'predictions': predictions,
            'stats': stats,
            'total': len(predictions)
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/model/status', methods=['GET'])
def model_status():
    """Get model status and metrics"""
    if not PREDICTION_ENABLED:
        return jsonify({
            'enabled': False,
            'message': 'Model not loaded'
        }), 200
    
    try:
        return jsonify({
            'enabled': True,
            'metrics': price_model.metrics,
            'countries': len(price_model.countries),
            'cities': sum(len(c) for c in price_model.cities_by_country.values())
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# API ENDPOINTS - Profile Management
# ============================================================================

# File upload configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/api/profile', methods=['PUT'])
@require_login
def update_profile():
    """Update user profile information"""
    try:
        user_id = session.get('user_id')
        data = request.get_json()
        
        update_data = {}
        allowed_profile_fields = ['full_name', 'phone', 'location', 'user_city', 'user_country']
        
        for field in allowed_profile_fields:
            if field in data:
                update_data[field] = data[field].strip() if data[field] else None
        
        if db.update_user(user_id, **update_data):
            updated_user = db.get_user_by_id(user_id)
            return jsonify({
                'message': 'Profile updated successfully',
                'user': user_to_dict(updated_user)
            }), 200
        else:
            return jsonify({'error': 'Failed to update profile'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/profile/picture', methods=['POST'])
@require_login
def upload_profile_picture():
    """Upload profile picture"""
    try:
        user_id = session.get('user_id')
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            # Create upload directory if not exists
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            
            # Generate unique filename
            ext = file.filename.rsplit('.', 1)[1].lower()
            filename = f"profile_{user_id}_{int(datetime.now().timestamp())}.{ext}"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            
            file.save(filepath)
            
            # Update user profile picture path
            profile_picture_url = f"/static/uploads/{filename}"
            db.update_user(user_id, profile_picture=profile_picture_url)
            
            return jsonify({
                'message': 'Profile picture uploaded successfully',
                'profile_picture': profile_picture_url
            }), 200
        else:
            return jsonify({'error': 'Invalid file type'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/profile/password', methods=['PUT'])
@require_login
def change_password():
    """Change user password"""
    try:
        user_id = session.get('user_id')
        data = request.get_json()
        
        if 'current_password' not in data or 'new_password' not in data:
            return jsonify({'error': 'Current and new password required'}), 400
        
        user = db.get_user_by_id(user_id)
        
        # Verify current password
        if not check_password_hash(user['password'], data['current_password']):
            return jsonify({'error': 'Current password is incorrect'}), 401
        
        # Validate new password
        if len(data['new_password']) < 6:
            return jsonify({'error': 'New password must be at least 6 characters'}), 400
        
        # Update password
        hashed_password = generate_password_hash(data['new_password'])
        if db.update_user(user_id, password=hashed_password):
            return jsonify({'message': 'Password changed successfully'}), 200
        else:
            return jsonify({'error': 'Failed to change password'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# API ENDPOINTS - Blog Section
# ============================================================================

@app.route('/blog')
def blog_page():
    """Serve blog page"""
    return render_template('blog.html')


# ============================================================================
# API ENDPOINTS - Contact Us
# ============================================================================

@app.route('/contact')
def contact_page():
    """Serve contact page"""
    return render_template('contact.html')


@app.route('/api/contact', methods=['POST'])
def submit_contact():
    """Handle contact form submission"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['name', 'email', 'subject', 'message']
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({'error': f'{field} is required'}), 400
        
        name = data['name'].strip()
        email = data['email'].strip().lower()
        subject = data['subject'].strip()
        message = data['message'].strip()
        
        # Simple email validation
        if '@' not in email:
            return jsonify({'error': 'Invalid email format'}), 400
        
        # Log the contact message (in production, you might save to DB or send email)
        print(f"\n{'='*50}")
        print("📧 New Contact Form Submission")
        print(f"{'='*50}")
        print(f"Name: {name}")
        print(f"Email: {email}")
        print(f"Subject: {subject}")
        print(f"Message: {message}")
        print(f"{'='*50}\n")
        
        return jsonify({
            'message': 'Contact form submitted successfully',
            'success': True
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# API ENDPOINTS - Real Estate News
# ============================================================================

@app.route('/api/news', methods=['GET'])
def get_real_estate_news():
    """Get latest real estate news and market updates"""
    
    news = [
        {
            'id': 1,
            'title': 'Housing Market Shows Signs of Recovery in Key Metro Areas',
            'summary': 'Major metropolitan areas are experiencing increased buyer activity as mortgage rates stabilize around 6.5%. Experts predict a balanced market for 2025.',
            'content': '''The housing market in major metropolitan areas across the country is showing encouraging signs of recovery as we approach 2025. After a challenging period of high mortgage rates that peaked above 7%, rates have now stabilized around 6.5%, sparking renewed buyer interest.

Key findings from our research include:

• **Increased Buyer Activity**: First-time homebuyers are returning to the market, with applications up 15% compared to the same period last year.

• **Price Stabilization**: While prices remain elevated in most markets, the rapid appreciation of 2021-2022 has moderated to more sustainable levels of 3-5% annual growth.

• **Inventory Improvements**: New listings are gradually increasing as homeowners who were reluctant to sell at lower prices begin to adjust to market realities.

• **Regional Variations**: The Sun Belt continues to attract buyers seeking affordability, while coastal metros see steady demand despite higher prices.

Experts predict that 2025 will bring a more balanced market, with neither buyers nor sellers having a decisive advantage. This equilibrium could create favorable conditions for those who have been waiting on the sidelines to enter the market.''',
            'category': 'Market Update',
            'date': 'Dec 26, 2024',
            'icon': '📈',
            'source': 'PropertyAI Research'
        },
        {
            'id': 2,
            'title': 'AI-Powered Valuations Becoming Industry Standard',
            'summary': 'Leading real estate firms are adopting AI valuation tools, with accuracy rates exceeding 95% in major markets. Traditional appraisals may become supplementary.',
            'content': '''The real estate industry is undergoing a technological transformation as AI-powered property valuation tools become increasingly mainstream. Major brokerages and financial institutions are now integrating these systems into their core operations.

**Key Developments in AI Valuation:**

• **Accuracy Improvements**: Modern AI valuation models now achieve accuracy rates exceeding 95% in well-documented markets, rivaling human appraisers.

• **Speed and Efficiency**: What once took days or weeks can now be completed in minutes, dramatically reducing transaction timelines.

• **Cost Savings**: Automated valuations typically cost a fraction of traditional appraisals, making them accessible for a wider range of use cases.

• **Data Integration**: AI systems can analyze thousands of data points including market trends, neighborhood characteristics, and property features simultaneously.

**Industry Impact:**

Traditional appraisers are adapting by focusing on complex or unique properties where human judgment remains essential. Many are incorporating AI tools into their workflow to enhance their own accuracy and efficiency.

Regulatory bodies are also taking notice, with several states now allowing AI valuations for certain transaction types, particularly refinancing and home equity applications.''',
            'category': 'Technology',
            'date': 'Dec 25, 2024',
            'icon': '🤖',
            'source': 'Tech Real Estate Weekly'
        },
        {
            'id': 3,
            'title': 'New Tax Benefits for First-Time Home Buyers in 2025',
            'summary': 'Government announces expanded tax credits for first-time buyers, including up to $15,000 in down payment assistance for qualified applicants.',
            'content': '''In a major policy announcement, the government has unveiled an expanded package of tax benefits aimed at helping first-time homebuyers enter the housing market in 2025.

**Key Program Features:**

• **Down Payment Assistance**: Qualified first-time buyers can receive up to $15,000 in down payment assistance through a refundable tax credit.

• **Income Eligibility**: The program is available to individuals earning up to $125,000 annually, or $200,000 for married couples filing jointly.

• **Property Requirements**: The purchased home must be the buyer's primary residence and meet local price limits based on area median home values.

• **Repayment Terms**: The assistance is structured as a forgivable loan, with no repayment required if the buyer remains in the home for at least 5 years.

**How to Apply:**

Applications will be processed through participating lenders starting January 1, 2025. Prospective buyers should begin gathering documentation now, including:

- Proof of first-time buyer status
- Income verification (tax returns, pay stubs)
- Pre-approval from a participating lender

Housing advocates estimate that this program could help an additional 500,000 families achieve homeownership in its first year.''',
            'category': 'Policy',
            'date': 'Dec 24, 2024',
            'icon': '🏛️',
            'source': 'Housing Policy Today'
        },
        {
            'id': 4,
            'title': 'Sustainable Homes Command 8% Premium in Urban Markets',
            'summary': 'Properties with solar panels, EV charging, and energy-efficient features are selling faster and at higher prices than comparable traditional homes.',
            'content': '''New research reveals that homes with sustainable features are commanding significant premiums in urban real estate markets, with buyers willing to pay an average of 8% more for eco-friendly properties.

**Most Valued Green Features:**

• **Solar Panel Systems**: Homes with owned (not leased) solar installations see the highest premiums, averaging 4-6% above comparable properties.

• **EV Charging Stations**: With electric vehicle adoption accelerating, Level 2 home chargers add an average of 2-3% to property values.

• **Energy-Efficient Windows and Insulation**: Properties with recent energy efficiency upgrades sell 12% faster than those without.

• **Smart Home Energy Management**: Integrated systems that optimize heating, cooling, and electricity use are increasingly sought after.

**Market Dynamics:**

The premium for sustainable homes is most pronounced in urban areas where environmentally conscious buyers concentrate. Cities with strong climate action plans, such as San Francisco, Seattle, and Boston, show the highest green premiums.

**Investment Considerations:**

For homeowners considering green upgrades, the data suggests strong returns on investment:

- Solar panels: 70-90% of installation cost recovered at sale
- Energy-efficient HVAC: 50-70% cost recovery
- Smart thermostats and monitoring: Near-complete cost recovery

These improvements also reduce monthly operating costs while the owner occupies the property.''',
            'category': 'Green Living',
            'date': 'Dec 23, 2024',
            'icon': '🌱',
            'source': 'Eco Property Report'
        },
        {
            'id': 5,
            'title': 'Remote Work Continues to Shape Suburban Demand',
            'summary': 'Suburban and secondary markets maintain strong demand as hybrid work becomes permanent for many companies. Home office space now top buyer priority.',
            'content': '''The shift to remote and hybrid work arrangements continues to reshape housing demand patterns, with suburban and secondary markets maintaining strength even as some workers return to offices.

**Key Trends:**

• **Home Office Priority**: 78% of current homebuyers rank dedicated home office space as a "must-have" feature, up from just 35% in 2019.

• **Space Over Location**: Buyers are increasingly willing to trade proximity to urban centers for larger homes with outdoor space and room for remote work.

• **Secondary Cities Rising**: Markets like Boise, Raleigh, and Austin continue to attract remote workers seeking affordability and quality of life.

**Hybrid Work Impact:**

With most companies settling on 2-3 days per week in-office arrangements, workers are optimizing for commute times that are acceptable on those days while prioritizing home comfort for remote days.

**What Buyers Are Seeking:**

1. **Dedicated office space** - Not just a bedroom with a desk, but purpose-built work areas
2. **High-speed internet availability** - Fiber connectivity is now a key selling point
3. **Outdoor amenities** - Private yards and community spaces for breaks and recreation
4. **Multi-functional spaces** - Rooms that can serve as office, gym, or classroom

Real estate agents report that highlighting work-from-home features in listings generates significantly more interest and faster sales in the current market.''',
            'category': 'Trends',
            'date': 'Dec 22, 2024',
            'icon': '🏡',
            'source': 'Work & Home Journal'
        }
    ]
    
    return jsonify({'news': news}), 200



# ============================================================================
# API ENDPOINTS - Analytics Dashboard
# ============================================================================

@app.route('/analytics')
def analytics_page():
    """Serve analytics dashboard"""
    if 'user_id' not in session:
        return redirect('/auth')
    return render_template('analytics.html')


@app.route('/api/analytics/stats', methods=['GET'])
def get_analytics_stats():
    """Get analytics statistics from real dataset"""
    try:
        import pandas as pd
        
        # Load the unified dataset
        df = pd.read_csv('outputs/unified_property_data.csv')
        
        # Basic KPIs
        total_properties = len(df)
        avg_price = df['price_usd'].mean()
        median_price = df['price_usd'].median()
        min_price = df['price_usd'].min()
        max_price = df['price_usd'].max()
        total_countries = df['country'].nunique()
        total_cities = df['city'].nunique()
        avg_area = df['area_sqm'].mean()
        avg_rooms = df['rooms'].mean()
        
        # Price by country
        price_by_country = df.groupby('country')['price_usd'].mean().round(2).to_dict()
        count_by_country = df['country'].value_counts().to_dict()
        
        # Price by city (top 20)
        city_stats = df.groupby(['city', 'country']).agg({
            'price_usd': ['mean', 'count'],
            'area_sqm': 'mean',
            'rooms': 'mean'
        }).reset_index()
        city_stats.columns = ['city', 'country', 'avg_price', 'count', 'avg_area', 'avg_rooms']
        city_stats = city_stats.nlargest(20, 'count')
        top_cities = city_stats.to_dict('records')
        
        # Price per sqm by country
        df['price_per_sqm'] = df['price_usd'] / df['area_sqm']
        price_per_sqm_by_country = df.groupby('country')['price_per_sqm'].mean().round(2).to_dict()
        
        # Room distribution
        room_distribution = df['rooms'].value_counts().sort_index().to_dict()
        
        # Furnishing distribution
        furnishing_dist = df['furnishing_status'].value_counts().to_dict()
        
        # Price ranges
        price_ranges = {
            'under_100k': len(df[df['price_usd'] < 100000]),
            '100k_500k': len(df[(df['price_usd'] >= 100000) & (df['price_usd'] < 500000)]),
            '500k_1m': len(df[(df['price_usd'] >= 500000) & (df['price_usd'] < 1000000)]),
            '1m_5m': len(df[(df['price_usd'] >= 1000000) & (df['price_usd'] < 5000000)]),
            'over_5m': len(df[df['price_usd'] >= 5000000])
        }
        
        # Area ranges
        area_ranges = {
            'under_50': len(df[df['area_sqm'] < 50]),
            '50_100': len(df[(df['area_sqm'] >= 50) & (df['area_sqm'] < 100)]),
            '100_200': len(df[(df['area_sqm'] >= 100) & (df['area_sqm'] < 200)]),
            '200_500': len(df[(df['area_sqm'] >= 200) & (df['area_sqm'] < 500)]),
            'over_500': len(df[df['area_sqm'] >= 500])
        }
        
        # Insights
        most_expensive_country = max(price_by_country, key=price_by_country.get)
        cheapest_country = min(price_by_country, key=price_by_country.get)
        most_properties_country = max(count_by_country, key=count_by_country.get)
        
        insights = [
            f"📊 Dataset contains {total_properties:,} properties across {total_countries} countries",
            f"💰 Average property price is ${avg_price:,.0f} USD",
            f"🏠 Most expensive market: {most_expensive_country} (${price_by_country[most_expensive_country]:,.0f} avg)",
            f"💵 Most affordable market: {cheapest_country} (${price_by_country[cheapest_country]:,.0f} avg)",
            f"📈 {most_properties_country} has the most listings ({count_by_country[most_properties_country]:,} properties)",
            f"📐 Average property size is {avg_area:.0f} sqm with {avg_rooms:.1f} rooms"
        ]
        
        return jsonify({
            'total_properties': total_properties,
            'avg_price': round(avg_price, 2),
            'median_price': round(median_price, 2),
            'min_price': round(min_price, 2),
            'max_price': round(max_price, 2),
            'total_countries': total_countries,
            'total_cities': total_cities,
            'avg_area': round(avg_area, 2),
            'avg_rooms': round(avg_rooms, 2),
            'price_by_country': price_by_country,
            'count_by_country': count_by_country,
            'price_per_sqm_by_country': price_per_sqm_by_country,
            'top_cities': top_cities,
            'room_distribution': room_distribution,
            'furnishing_distribution': furnishing_dist,
            'price_ranges': price_ranges,
            'area_ranges': area_ranges,
            'insights': insights
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/analytics/heatmap', methods=['GET'])
def get_heatmap_data():
    """Get heatmap data from real dataset with city coordinates"""
    try:
        import pandas as pd
        
        # City coordinates (major cities from the dataset)
        city_coords = {
            # India
            'Mumbai': (19.0760, 72.8777), 'Delhi': (28.6139, 77.2090), 'Bangalore': (12.9716, 77.5946),
            'Hyderabad': (17.3850, 78.4867), 'Chennai': (13.0827, 80.2707), 'Kolkata': (22.5726, 88.3639),
            'Pune': (18.5204, 73.8567), 'Ahmedabad': (23.0225, 72.5714), 'Jaipur': (26.9124, 75.7873),
            'Lucknow': (26.8467, 80.9462), 'Surat': (21.1702, 72.8311), 'Indore': (22.7196, 75.8577),
            # USA
            'New York': (40.7128, -74.0060), 'Los Angeles': (34.0522, -118.2437), 'Chicago': (41.8781, -87.6298),
            'Houston': (29.7604, -95.3698), 'Phoenix': (33.4484, -112.0740), 'Philadelphia': (39.9526, -75.1652),
            'San Antonio': (29.4241, -98.4936), 'San Diego': (32.7157, -117.1611), 'Dallas': (32.7767, -96.7970),
            'San Jose': (37.3382, -121.8863), 'Austin': (30.2672, -97.7431), 'Seattle': (47.6062, -122.3321),
            'Boston': (42.3601, -71.0589), 'Denver': (39.7392, -104.9903), 'Miami': (25.7617, -80.1918),
            # Japan
            'Tokyo': (35.6762, 139.6503), 'Osaka': (34.6937, 135.5023), 'Yokohama': (35.4437, 139.6380),
            'Nagoya': (35.1815, 136.9066), 'Sapporo': (43.0618, 141.3545), 'Kobe': (34.6901, 135.1956),
            # Poland
            'Warsaw': (52.2297, 21.0122), 'Krakow': (50.0647, 19.9450), 'Wroclaw': (51.1079, 17.0385),
            'Gdansk': (54.3520, 18.6466), 'Poznan': (52.4064, 16.9252), 'Lodz': (51.7592, 19.4560),
            # Bangladesh
            'Dhaka': (23.8103, 90.4125), 'Chittagong': (22.3569, 91.7832), 'Khulna': (22.8456, 89.5403),
        }
        
        df = pd.read_csv('outputs/unified_property_data.csv')
        
        # Aggregate by city
        city_stats = df.groupby(['city', 'country']).agg({
            'price_usd': 'mean',
            'area_sqm': 'mean',
            'rooms': 'mean'
        }).reset_index()
        city_stats.columns = ['city', 'country', 'avg_price', 'avg_area', 'avg_rooms']
        
        # Add count
        city_counts = df.groupby('city').size().reset_index(name='property_count')
        city_stats = city_stats.merge(city_counts, on='city')
        
        heatmap_data = []
        for _, row in city_stats.iterrows():
            city_name = row['city']
            if city_name in city_coords:
                lat, lng = city_coords[city_name]
                heatmap_data.append({
                    'lat': lat,
                    'lng': lng,
                    'city': city_name,
                    'country': row['country'],
                    'avg_price': round(row['avg_price'], 0),
                    'avg_area': round(row['avg_area'], 1),
                    'avg_rooms': round(row['avg_rooms'], 1),
                    'property_count': int(row['property_count'])
                })
        
        return jsonify({'heatmap_data': heatmap_data}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# API ENDPOINTS - Location Intelligence
# ============================================================================

# Import location intelligence modules
try:
    from location_database import LocationDatabase
    from location_intelligence import LocationIntelligence
    location_db = LocationDatabase()
    location_engine = LocationIntelligence(location_db)
    LOCATION_ENABLED = True
    print("✓ Location Intelligence engine initialized")
except Exception as e:
    LOCATION_ENABLED = False
    location_db = None
    location_engine = None
    print(f"⚠ Location Intelligence not available: {e}")


@app.route('/location-map')
def location_map_page():
    """Serve the interactive location map page"""
    if 'user_id' not in session:
        return redirect('/auth')
    return render_template('location_map.html')


@app.route('/api/location/heatmap', methods=['GET'])
def get_location_heatmap():
    """
    Get price heatmap data using Kernel Density Estimation.
    Returns intensity values for visualization with color gradient.
    """
    if not LOCATION_ENABLED:
        return jsonify({'error': 'Location service not available'}), 503
    
    try:
        # Get filter parameters
        city = request.args.get('city')
        country = request.args.get('country')
        
        # Get property coordinates
        coords = location_db.get_property_coordinates(city=city, country=country)
        
        if not coords:
            # Fallback to all cities with coords
            coords = location_db.get_all_cities_with_coords()
        
        # If still empty, return sample data
        if not coords:
            # Use sample data from existing analytics heatmap
            import pandas as pd
            
            city_coords_map = {
                'Mumbai': (19.0760, 72.8777), 'Delhi': (28.6139, 77.2090), 
                'Bangalore': (12.9716, 77.5946), 'Hyderabad': (17.3850, 78.4867),
                'Chennai': (13.0827, 80.2707), 'Pune': (18.5204, 73.8567),
                'Tokyo': (35.6762, 139.6503), 'Osaka': (34.6937, 135.5023),
                'Warsaw': (52.2297, 21.0122), 'Krakow': (50.0647, 19.9450),
                'Dhaka': (23.8103, 90.4125), 'New York': (40.7128, -74.0060),
                'Los Angeles': (34.0522, -118.2437)
            }
            
            df = pd.read_csv('outputs/unified_property_data.csv')
            city_stats = df.groupby(['city', 'country']).agg({
                'price_usd': 'mean',
                'area_sqm': 'mean'
            }).reset_index()
            
            coords = []
            for _, row in city_stats.iterrows():
                if row['city'] in city_coords_map:
                    lat, lng = city_coords_map[row['city']]
                    coords.append({
                        'latitude': lat,
                        'longitude': lng,
                        'city': row['city'],
                        'country': row['country'],
                        'price_usd': row['price_usd'],
                        'area_sqm': row['area_sqm']
                    })
        
        # Compute heatmap
        heatmap_points = location_engine.compute_price_heatmap(coords)
        
        # Get city-level stats for markers
        city_stats = []
        for coord in coords:
            if 'city' in coord:
                price = coord.get('avg_price_sqft', coord.get('price_usd', 0) / max(coord.get('area_sqm', 1), 1))
                city_stats.append({
                    'lat': coord['latitude'],
                    'lng': coord['longitude'],
                    'city': coord.get('city', 'Unknown'),
                    'country': coord.get('country', ''),
                    'avg_price_sqft': round(price, 0),
                    'property_count': coord.get('property_count', coord.get('total_properties', 0)),
                    'intensity': min(1.0, price / 50000),
                    'color': location_engine.get_heat_intensity_color(min(1.0, price / 50000))
                })
        
        return jsonify({
            'heatmap_points': heatmap_points[:500],  # Limit for performance
            'city_markers': city_stats,
            'legend': {
                'low': '#0000ff',
                'medium': '#ffff00',
                'high': '#ff0000',
                'description': 'Blue = Low price | Yellow = Medium | Red = High'
            }
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/location/proximity', methods=['GET'])
def get_location_proximity():
    """
    Calculate proximity score and distances to nearby POIs.
    
    Query params:
        lat: latitude
        lng: longitude
        city: optional city filter
    
    Returns:
        proximity_score (0-100), individual distances, ratings
    """
    if not LOCATION_ENABLED:
        return jsonify({'error': 'Location service not available'}), 503
    
    try:
        lat = request.args.get('lat', type=float)
        lng = request.args.get('lng', type=float)
        city = request.args.get('city')
        
        if lat is None or lng is None:
            return jsonify({'error': 'lat and lng parameters required'}), 400
        
        # Get nearby POIs
        pois = location_db.get_nearby_pois(lat, lng, radius_km=10)
        
        if not pois:
            # Get POIs by city if available
            if city:
                pois = []
                for poi_type in ['metro', 'school', 'hospital', 'it_park', 'commercial']:
                    pois.extend(location_db.get_pois_by_type(poi_type, city))
        
        # Calculate proximity score
        result = location_engine.calculate_proximity_score(lat, lng, pois)
        
        return jsonify({
            'latitude': lat,
            'longitude': lng,
            'proximity_score': result['proximity_score'],
            'rating': result['rating'],
            'distances': result['distances'],
            'nearby_pois_count': len(pois)
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/location/demand-supply', methods=['GET'])
def get_demand_supply():
    """
    Get demand-supply analytics for a location.
    
    Query params:
        lat: latitude
        lng: longitude
        city: city name
    
    Returns:
        demand_score, supply_score, ratio, market status
    """
    if not LOCATION_ENABLED:
        return jsonify({'error': 'Location service not available'}), 503
    
    try:
        lat = request.args.get('lat', type=float)
        lng = request.args.get('lng', type=float)
        city = request.args.get('city')
        
        if not city and (lat is None or lng is None):
            return jsonify({'error': 'city or lat/lng parameters required'}), 400
        
        # Get demand-supply metrics
        metrics = location_db.get_demand_supply_metrics(city) if city else None
        
        if metrics:
            demand_data = {
                'search_frequency': metrics.get('search_frequency', 0),
                'prediction_requests': metrics.get('prediction_requests', 0),
                'interaction_count': metrics.get('interaction_count', 0)
            }
            supply_data = {
                'active_listings': metrics.get('active_listings', 0),
                'new_developments': metrics.get('new_developments', 0)
            }
        else:
            # Default values
            demand_data = {'search_frequency': 500, 'prediction_requests': 150, 'interaction_count': 1000}
            supply_data = {'active_listings': 200, 'new_developments': 15}
        
        result = location_engine.calculate_demand_supply(demand_data, supply_data)
        
        return jsonify({
            'latitude': lat,
            'longitude': lng,
            'city': city,
            'demand_score': result['demand_score'],
            'supply_score': result['supply_score'],
            'demand_supply_ratio': result['demand_supply_ratio'],
            'market_status': result['market_status'],
            'status_color': result['status_color'],
            'visualization': result['visualization']
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/location/infrastructure', methods=['GET'])
def get_infrastructure_score():
    """
    Get infrastructure development score for a location.
    
    Query params:
        lat: latitude
        lng: longitude
        city: optional city filter
    
    Returns:
        infrastructure_score (0-100), nearby projects, appreciation potential
    """
    if not LOCATION_ENABLED:
        return jsonify({'error': 'Location service not available'}), 503
    
    try:
        lat = request.args.get('lat', type=float)
        lng = request.args.get('lng', type=float)
        city = request.args.get('city')
        
        if lat is None or lng is None:
            return jsonify({'error': 'lat and lng parameters required'}), 400
        
        # Get infrastructure projects
        projects = location_db.get_infrastructure_projects(city=city)
        
        # Calculate score
        result = location_engine.calculate_infrastructure_score(lat, lng, projects)
        
        return jsonify({
            'latitude': lat,
            'longitude': lng,
            'infrastructure_score': result['infrastructure_score'],
            'appreciation_potential': result['appreciation_potential'],
            'nearby_projects': result['nearby_projects'],
            'total_projects': result['total_projects']
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/location/safety-livability', methods=['GET'])
def get_safety_livability():
    """
    Get safety and livability indices for a location.
    
    Query params:
        lat: latitude
        lng: longitude
        city: city name
        area: optional area/neighborhood
    
    Returns:
        safety_index (0-100), livability_index (0-100), breakdown
    """
    if not LOCATION_ENABLED:
        return jsonify({'error': 'Location service not available'}), 503
    
    try:
        lat = request.args.get('lat', type=float)
        lng = request.args.get('lng', type=float)
        city = request.args.get('city')
        area = request.args.get('area')
        
        if not city:
            return jsonify({'error': 'city parameter required'}), 400
        
        # Get safety/livability data
        data = location_db.get_safety_livability(city, area)
        
        if data and len(data) > 0:
            # Use the first matching record
            record = data[0]
            result = location_engine.calculate_safety_livability(
                crime_index=record.get('crime_index', 50),
                traffic_index=record.get('traffic_index', 50),
                aqi=record.get('aqi', 100),
                green_space=record.get('green_space_score', 50),
                noise_level=record.get('noise_level', 50)
            )
        else:
            # Default values
            result = location_engine.calculate_safety_livability()
        
        return jsonify({
            'latitude': lat,
            'longitude': lng,
            'city': city,
            'area': area,
            'safety_index': result['safety_index'],
            'safety_rating': result['safety_rating'],
            'livability_index': result['livability_index'],
            'livability_rating': result['livability_rating'],
            'breakdown': result['breakdown']
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/location/features', methods=['GET'])
def get_location_features():
    """
    Get complete AI-ready location feature vector.
    
    Query params:
        lat: latitude
        lng: longitude
        city: city name
        country: country name
    
    Returns:
        Complete feature vector for ML models
    """
    if not LOCATION_ENABLED:
        return jsonify({'error': 'Location service not available'}), 503
    
    try:
        lat = request.args.get('lat', type=float)
        lng = request.args.get('lng', type=float)
        city = request.args.get('city')
        country = request.args.get('country')
        
        if lat is None or lng is None:
            return jsonify({'error': 'lat and lng parameters required'}), 400
        
        # Check cache first
        cached = location_db.get_location_features(lat, lng)
        if cached:
            return jsonify({
                'features': cached,
                'cached': True
            }), 200
        
        # Get all required data
        pois = location_db.get_nearby_pois(lat, lng, radius_km=10)
        projects = location_db.get_infrastructure_projects(city=city)
        
        # Get safety data
        safety_data = None
        if city:
            safety_records = location_db.get_safety_livability(city)
            if safety_records:
                record = safety_records[0]
                safety_data = {
                    'crime_index': record.get('crime_index', 50),
                    'traffic_index': record.get('traffic_index', 50),
                    'aqi': record.get('aqi', 100),
                    'green_space': record.get('green_space_score', 50),
                    'noise_level': record.get('noise_level', 50)
                }
        
        # Get demand-supply data
        demand_data = None
        supply_data = None
        if city:
            metrics = location_db.get_demand_supply_metrics(city)
            if metrics:
                demand_data = {
                    'search_frequency': metrics.get('search_frequency', 0),
                    'prediction_requests': metrics.get('prediction_requests', 0),
                    'interaction_count': metrics.get('interaction_count', 0)
                }
                supply_data = {
                    'active_listings': metrics.get('active_listings', 0),
                    'new_developments': metrics.get('new_developments', 0)
                }
        
        # Get avg price
        coords = location_db.get_property_coordinates(city=city)
        avg_price_sqft = coords[0].get('avg_price_sqft', 0) if coords else 0
        
        # Generate features
        features = location_engine.get_location_features(
            lat=lat,
            lng=lng,
            city=city,
            country=country,
            pois=pois,
            projects=projects,
            safety_data=safety_data,
            demand_data=demand_data,
            supply_data=supply_data,
            avg_price_sqft=avg_price_sqft
        )
        
        # Cache the features
        location_db.save_location_features(lat, lng, features)
        
        return jsonify({
            'features': features,
            'cached': False
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/location/impact', methods=['POST'])
def get_location_impact():
    """
    Get explainable AI location impact for price prediction.
    
    Body:
        lat, lng: coordinates
        base_price: base predicted price
        city, country: location identifiers
    
    Returns:
        Location impact breakdown with percentage contributions
    """
    if not LOCATION_ENABLED:
        return jsonify({'error': 'Location service not available'}), 503
    
    try:
        data = request.get_json()
        
        lat = data.get('lat')
        lng = data.get('lng')
        base_price = data.get('base_price', 0)
        city = data.get('city')
        country = data.get('country')
        
        if lat is None or lng is None:
            return jsonify({'error': 'lat and lng required'}), 400
        
        # Get or compute features
        cached = location_db.get_location_features(lat, lng)
        
        if not cached:
            # Get features (simplified)
            features = location_engine.get_location_features(
                lat=lat, lng=lng, city=city, country=country
            )
        else:
            features = cached
        
        # Calculate impact
        impact = location_engine.get_location_impact_explanation(features, base_price)
        
        return jsonify(impact), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/location/cities', methods=['GET'])
def get_location_cities():
    """Get all cities with location data"""
    if not LOCATION_ENABLED:
        return jsonify({'error': 'Location service not available'}), 503
    
    try:
        cities = location_db.get_all_cities_with_coords()
        return jsonify({'cities': cities}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Admin endpoints for location data management
@app.route('/api/admin/location/refresh', methods=['POST'])
@require_login
def refresh_location_features():
    """Trigger recalculation of all location features (Admin only)"""
    try:
        current_user = db.get_user_by_id(session.get('user_id'))
        if current_user['role'] != 'admin':
            return jsonify({'error': 'Admin access required'}), 403
        
        if not LOCATION_ENABLED:
            return jsonify({'error': 'Location service not available'}), 503
        
        # Get all coordinates and recompute features
        coords = location_db.get_all_cities_with_coords()
        refreshed = 0
        
        for coord in coords:
            features = location_engine.get_location_features(
                lat=coord['latitude'],
                lng=coord['longitude'],
                city=coord.get('city'),
                country=coord.get('country')
            )
            location_db.save_location_features(coord['latitude'], coord['longitude'], features)
            refreshed += 1
        
        return jsonify({
            'message': f'Refreshed {refreshed} location features',
            'count': refreshed
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/admin/location/seed', methods=['POST'])
@require_login
def seed_location_data():
    """Seed the database with sample location data (Admin only)"""
    try:
        current_user = db.get_user_by_id(session.get('user_id'))
        if current_user['role'] != 'admin':
            return jsonify({'error': 'Admin access required'}), 403
        
        # Import and run seeder
        from location_data_seeder import seed_location_data as run_seeder
        run_seeder()
        
        return jsonify({
            'message': 'Location data seeded successfully'
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# API ENDPOINTS - Blog Content
# ============================================================================

@app.route('/api/blogs', methods=['GET'])
def get_blogs():

    """Get all blog articles with complete content"""
    blogs = [
        {
            'id': 1,
            'title': 'How AI is Revolutionizing Real Estate Valuation',
            'category': 'Technology',
            'image': 'https://images.unsplash.com/photo-1560472354-b33ff0c44a43?w=800',
            'excerpt': 'Discover how artificial intelligence is transforming property valuation with unprecedented accuracy and speed.',
            'author': 'Dr. Sarah Chen',
            'date': 'Dec 15, 2024',
            'readTime': 8,
            'content': """Artificial Intelligence is fundamentally changing how we value real estate, bringing unprecedented accuracy and efficiency to an industry traditionally reliant on manual assessments and subjective judgments.

THE EVOLUTION OF PROPERTY VALUATION

For decades, property valuation relied heavily on comparable sales analysis, where appraisers would manually compare a property to recently sold homes in the area. While effective, this method had limitations: it was time-consuming, subject to human bias, and often struggled with unique properties or rapidly changing markets.

Enter AI-powered valuation models. These sophisticated systems analyze millions of data points in seconds, considering factors that human appraisers might overlook or find difficult to quantify.

HOW AI VALUATION WORKS

Modern AI valuation systems use machine learning algorithms trained on vast datasets of property transactions. These models consider:

1. Physical Characteristics: Square footage, bedrooms, bathrooms, lot size, construction year, and building materials.

2. Location Intelligence: Proximity to schools, hospitals, shopping centers, public transportation, and employment hubs. AI can even factor in noise levels, crime statistics, and walkability scores.

3. Market Dynamics: Real-time analysis of supply and demand, days on market, price trends, and seasonal patterns.

4. Economic Indicators: Interest rates, employment rates, population growth, and local economic development projects.

5. Visual Analysis: Advanced computer vision algorithms can assess property condition from photos, detecting features like renovated kitchens, pool quality, or needed repairs.

ACCURACY AND RELIABILITY

Our PropertyAI system achieves over 95% accuracy within a 5-10% margin of the final sale price. This accuracy comes from:

- Training on 200,000+ verified property transactions across 5 countries
- Continuous learning from new sales data
- Cross-validation with multiple model architectures
- Integration of real-time market feeds

BENEFITS FOR BUYERS AND SELLERS

For Buyers:
- Instant valuations help set realistic expectations
- Identify underpriced properties before they're snapped up
- Make confident offers backed by AI analysis
- Avoid overpaying in competitive markets

For Sellers:
- Optimal pricing strategies to maximize returns
- Understand exactly what drives your property's value
- Time the market with predictive analytics
- Reduce days on market with accurate pricing

THE FUTURE OF AI IN REAL ESTATE

We're just scratching the surface. Future developments include:
- Predictive maintenance recommendations
- Investment return forecasting over 10-20 year horizons
- Automated negotiation support
- Integration with smart home data for condition monitoring

AI doesn't replace human expertise—it enhances it. The best results come from combining AI-powered insights with the nuanced understanding of experienced real estate professionals.

At PropertyAI, we're committed to making these powerful tools accessible to everyone, democratizing access to institutional-grade property analytics."""
        },
        {
            'id': 2,
            'title': 'First-Time Home Buyer\'s Complete Guide for 2025',
            'category': 'Guide',
            'image': 'https://images.unsplash.com/photo-1560518883-ce09059eeffa?w=800',
            'excerpt': 'Everything you need to know about buying your first home, from saving for a down payment to closing the deal.',
            'author': 'Michael Rodriguez',
            'date': 'Dec 12, 2024',
            'readTime': 12,
            'content': """Buying your first home is one of life's most exciting milestones—and one of the most complex financial decisions you'll ever make. This comprehensive guide walks you through every step of the journey.

STEP 1: ASSESS YOUR FINANCIAL READINESS

Before you start browsing listings, take an honest look at your finances:

Credit Score: Aim for at least 680 for conventional loans, though FHA loans accept scores as low as 580. Check your score 6-12 months before buying to address any issues.

Debt-to-Income Ratio: Lenders prefer your total monthly debt payments (including the new mortgage) to be under 43% of gross income. Calculate yours: (Total Monthly Debts ÷ Gross Monthly Income) × 100

Emergency Fund: Beyond your down payment, maintain 3-6 months of expenses for unexpected costs and the inevitable repairs that come with homeownership.

STEP 2: SAVE FOR YOUR DOWN PAYMENT

The 20% down payment is traditional but not mandatory:
- Conventional loans: As low as 3%
- FHA loans: 3.5%
- VA loans: 0% for eligible veterans
- USDA loans: 0% for rural properties

However, putting down less than 20% typically requires Private Mortgage Insurance (PMI), adding $50-$200+ monthly to your payment.

Saving strategies:
- Automate transfers to a dedicated savings account
- Cut subscriptions you don't use
- Consider a side income source
- Look into first-time buyer assistance programs in your area

STEP 3: GET PRE-APPROVED FOR A MORTGAGE

Pre-approval shows sellers you're serious and tells you exactly how much house you can afford. You'll need:
- W-2s and tax returns (2 years)
- Pay stubs (2 months)
- Bank statements (2 months)
- ID and Social Security number

Shop multiple lenders—rates can vary by 0.5% or more, translating to thousands over the loan's life.

STEP 4: FIND YOUR HOME

Work with a buyer's agent (their commission is typically paid by the seller). Consider:

Must-Haves vs. Nice-to-Haves: List non-negotiables (bedrooms, location, school district) separately from preferences (pool, updated kitchen).

Future Needs: Plan for 5-10 years ahead. Will you need more space for a growing family? A home office?

Commute: Test the commute during rush hour before committing.

Neighborhood: Visit at different times—evenings, weekends—to get a true feel.

STEP 5: MAKE AN OFFER

Your agent will help you craft a competitive offer based on:
- Comparable sales in the area
- How long the home has been listed
- Current market conditions (buyer's vs. seller's market)
- Any issues identified during viewings

Include contingencies for financing, inspection, and appraisal to protect yourself.

STEP 6: HOME INSPECTION

Never skip this. A professional inspector (budget $300-$500) examines:
- Structural integrity
- Roof condition
- Electrical and plumbing systems
- HVAC functionality
- Signs of water damage or mold
- Pest infestations

If major issues arise, you can renegotiate, request repairs, or walk away.

STEP 7: CLOSING

The final stretch involves:
- Final walkthrough (ensure repairs were made and nothing's changed)
- Reviewing the Closing Disclosure (your final loan terms and costs)
- Signing approximately 100 pages of documents
- Paying closing costs (typically 2-5% of purchase price)

FIRST-YEAR HOMEOWNER TIPS

1. Build a maintenance fund (1-2% of home value annually)
2. Change locks immediately
3. Locate your main water shutoff valve
4. Don't rush into renovations—live in the space first
5. Keep all receipts for potential tax deductions

Congratulations! You're ready to join the ranks of homeowners. Use PropertyAI's tools to find the perfect property and ensure you're paying a fair price."""
        },
        {
            'id': 3,
            'title': '5 Smart Real Estate Investment Strategies for Building Wealth',
            'category': 'Investment',
            'image': 'https://images.unsplash.com/photo-1560520031-3a4dc4e9de0c?w=800',
            'excerpt': 'Learn proven strategies for building long-term wealth through strategic real estate investments.',
            'author': 'Jennifer Park, CFA',
            'date': 'Dec 10, 2024',
            'readTime': 10,
            'content': """Real estate has created more millionaires than any other asset class. Here are five proven strategies to build wealth through property investment.

STRATEGY 1: BUY AND HOLD RENTAL PROPERTIES

The classic wealth-building approach: purchase properties, rent them out, and let time work its magic.

How it works:
- Tenants pay your mortgage while you build equity
- Properties appreciate over time (historically 3-5% annually)
- Rental income typically increases with inflation
- Significant tax advantages (depreciation, expense deductions)

The Numbers:
Let's say you buy a $300,000 property with 25% down ($75,000):
- Monthly rent: $2,000
- Mortgage payment: $1,400
- Cash flow: $400/month after expenses
- 10-year appreciation (at 4%/year): $144,000
- Equity from mortgage paydown: ~$50,000

Your $75,000 investment becomes $269,000 in equity—a 258% return, not counting cash flow.

STRATEGY 2: HOUSE HACKING

Perfect for beginners: buy a multi-unit property, live in one unit, rent the others.

Example: Purchase a duplex for $400,000
- Live in one unit
- Rent the other for $1,800/month
- Your effective housing cost: Dramatically reduced or free!

This lets you:
- Qualify for owner-occupied loan rates (lower than investment property rates)
- Learn landlording with training wheels
- Build equity while saving on rent
- Transition to full rental when you move

STRATEGY 3: BRRRR METHOD

Buy, Rehab, Rent, Refinance, Repeat. This strategy recycles your capital:

1. Buy a distressed property below market value
2. Rehab it to increase value significantly
3. Rent it out at market rates
4. Refinance based on new, higher value
5. Pull out your original investment + some profit
6. Repeat with the recycled capital

Example:
- Buy distressed: $150,000
- Rehab: $40,000
- After-repair value: $250,000
- Refinance at 75% LTV: $187,500
- Get back: $187,500 - $190,000 costs = initial investment recovered
- Now you have a cash-flowing property with almost nothing still invested!

STRATEGY 4: SHORT-TERM RENTALS (AIRBNB)

In the right locations, short-term rentals can generate 2-3x traditional rental income.

Best candidates:
- Tourist destinations
- Near convention centers or hospitals
- College towns (graduation weekends, game days)
- Business travel hubs

Considerations:
- Higher income but more work (guest turnover, cleaning)
- Local regulations vary—research before buying
- More exposure to economic downturns (travel is discretionary)
- Furnishing and setup costs

Use PropertyAI's ROI calculator to compare short-term vs. long-term rental potential.

STRATEGY 5: REAL ESTATE SYNDICATIONS

Pool money with other investors to buy larger properties you couldn't afford alone.

Benefits:
- Access to commercial properties, apartment complexes, development projects
- Passive income (general partners handle management)
- Diversification across multiple properties
- Entry points often $25,000-$100,000

Risks:
- Less control than direct ownership
- Illiquid (typically 5-7 year hold periods)
- Returns depend on syndicator's competence
- Due diligence critical

CALCULATING YOUR RETURNS

Key metrics every investor should know:

Cash-on-Cash Return: (Annual Cash Flow ÷ Cash Invested) × 100
Target: 8-12%

Cap Rate: (Net Operating Income ÷ Property Value) × 100
Varies by market and property type. 5-8% is common.

Total ROI: Include appreciation, equity paydown, cash flow, and tax benefits.

GETTING STARTED

1. Educate yourself (books, podcasts, this blog!)
2. Analyze 100 deals to calibrate your judgement
3. Build your team (agent, lender, inspector, contractor)
4. Start with one property and learn
5. Scale gradually as you gain experience

Remember: Real estate investing is a marathon, not a sprint. The wealthy build portfolios over decades. Start today, stay patient, and let compound growth work its magic."""
        },
        {
            'id': 4,
            'title': 'Real Estate Market Trends to Watch in 2025',
            'category': 'Market Analysis',
            'image': 'https://images.unsplash.com/photo-1551836022-4c4c79ecde36?w=800',
            'excerpt': 'Expert analysis of emerging trends shaping the global real estate market in 2025.',
            'author': 'Dr. Robert Yamamoto',
            'date': 'Dec 8, 2024',
            'readTime': 7,
            'content': """The real estate market in 2025 is being shaped by shifting demographics, technological innovation, and evolving work patterns. Here's what savvy investors and homebuyers need to know.

TREND 1: THE REMOTE WORK REVOLUTION CONTINUES

The pandemic permanently changed where people want to live. In 2025:

Winners:
- Suburban markets with good schools and outdoor space
- Secondary cities with lower costs and high quality of life (Austin, Boise, Raleigh)
- Areas with fast internet infrastructure
- Locations within 2-3 hours of major metros (the "Zoom town" effect)

Losers:
- Small urban apartments designed for commuters
- Expensive metros without quality-of-life advantages
- Areas with poor remote work infrastructure

Investment implication: Look beyond traditional "hot" markets. Cities offering lifestyle benefits at affordable prices are seeing sustained demand.

TREND 2: MORTGAGE RATE NORMALIZATION

After the volatility of 2022-2024, mortgage rates are stabilizing in the 5.5-6.5% range. This "new normal" means:

- Buyers have adjusted expectations
- The "lock-in effect" is easing (homeowners who refinanced at 3% are more willing to move)
- Inventory is gradually increasing
- Price growth is moderating to sustainable 3-5% annually

Strategy: Stop waiting for 3% rates—they're not coming back anytime soon. Today's rates are historically reasonable. Focus on finding the right property rather than timing the market.

TREND 3: GREEN BUILDINGS COMMAND PREMIUM

Energy efficiency has evolved from nice-to-have to must-have:

- Homes with solar panels sell for 4-6% more
- High-efficiency HVAC, windows, and insulation are major selling points
- EV charging capability is increasingly expected
- Green certifications (LEED, Energy Star) accelerate sales

Driver: Rising energy costs and climate awareness, plus younger buyers' environmental priorities.

Investment tip: When evaluating properties, calculate potential energy savings. A $15,000 solar installation might add $25,000 to home value while cutting $3,000/year in electricity costs.

TREND 4: MULTI-GENERATIONAL LIVING GROWS

Economic pressures and cultural shifts are driving demand for larger homes that accommodate:
- Adult children returning home
- Aging parents needing care
- Home offices for multiple remote workers
- Rental units for income (ADUs, mother-in-law suites)

Hot features:
- Separate entrances
- Two master suites
- Kitchenettes in basement/garage conversions
- Properties zoned for accessory dwelling units (ADUs)

TREND 5: INSTITUTIONAL INVESTORS RESHAPE MARKETS

Large investors (Blackstone, American Homes 4 Rent) continue buying single-family homes:

Impact:
- Competition for starter homes in desirable areas
- More professionally managed rental options
- Build-to-rent communities growing
- Higher standards for rental properties

What this means for you: Move quickly when you find the right property. Have your financing lined up. In competitive markets, cash or near-cash offers have significant advantages.

TREND 6: AI TRANSFORMS THE TRANSACTION

Technology is streamlining every step:
- AI-powered valuations (like PropertyAI) give instant, accurate pricing
- Virtual tours reduce unnecessary viewings
- Digital closings become standard
- Predictive analytics identify investment opportunities

Early adopters of these tools gain significant advantages in speed and decision-making quality.

REGIONAL SPOTLIGHT

Hottest Markets for 2025:
1. Austin, TX - Tech growth continues despite cooldown
2. Nashville, TN - Healthcare, music, and corporate relocations
3. Phoenix, AZ - Affordability refugee destination
4. Raleigh-Durham, NC - Research Triangle boom
5. Tampa, FL - Migration from Northeast continues

Watch Markets:
- Remote work may boost: Vermont, Maine, Montana
- Undervalued potential: Cincinnati, Pittsburgh, Buffalo

KEY TAKEAWAYS

1. Remote work has permanently changed geographic demand
2. Don't wait for ultra-low rates—they're not returning
3. Energy efficiency increasingly impacts value
4. Multi-generational layouts are smart long-term plays
5. Use technology (AI valuation, virtual tours) to gain competitive edge

The investors who thrive in 2025 will be those who understand these shifts and position themselves accordingly. Use PropertyAI's analytics to identify opportunities aligned with these trends."""
        },
        {
            'id': 5,
            'title': 'Smart Home Technology: What Adds Value vs. What Doesn\'t',
            'category': 'Home Improvement',
            'image': 'https://images.unsplash.com/photo-1558002038-1055907df827?w=800',
            'excerpt': 'Learn which smart home upgrades increase property value and which are just expensive gimmicks.',
            'author': 'Amanda Foster',
            'date': 'Dec 5, 2024',
            'readTime': 6,
            'content': """Smart home technology is everywhere, but not all upgrades are created equal when it comes to adding property value. Here's what actually pays off—and what's just a fancy expense.

HIGH-VALUE SMART HOME UPGRADES

1. Smart Thermostats (ROI: Excellent)
Cost: $150-$300
Value added: $500-$1,500
Why it works: Energy savings are tangible and verified. Buyers love the "set it and forget it" convenience. Nest, Ecobee, and Honeywell are recognized brands that signal quality.

2. Smart Security Systems (ROI: Very Good)
Cost: $200-$800
Value added: $1,000-$3,000
Why it works: Security is a universal priority. Video doorbells (Ring, Nest), smart locks, and integrated monitoring systems provide peace of mind. Insurance discounts add ongoing savings.

3. Smart Lighting (ROI: Good)
Cost: $100-$500
Value added: $300-$1,500
Why it works: Easy to understand and impressive in showings. Programmable scenes, remote control, and energy efficiency appeal broadly. Focus on switches/dimmers rather than just smart bulbs.

4. Motorized Window Treatments (ROI: Good)
Cost: $500-$2,000 per room
Value added: $1,000-$4,000 per room
Why it works: Perceived as luxury, particularly effective in homes with large or hard-to-reach windows. Energy efficiency angle (automated sun management) adds practical appeal.

5. Smart Garage Door Openers (ROI: Good)
Cost: $250-$400
Value added: $400-$800
Why it works: Simple utility—check if the door is closed from anywhere, let in deliveries, grant temporary access. Universal appeal with minimal learning curve.

MODERATE VALUE UPGRADES

Smart Irrigation Systems
Cost: $200-$500
Worth it if: You have significant landscaping to maintain
Considerations: Water savings can be documented; appeals to environmentally conscious buyers

Whole-Home WiFi Systems
Cost: $300-$600
Worth it if: The home has WiFi dead zones
Considerations: Increasingly expected by remote workers; mesh systems like Eero, Google Nest, or Ubiquiti impress tech-savvy buyers

Smart Smoke/CO Detectors
Cost: $100-$300
Worth it if: You're already upgrading other smart systems
Considerations: Safety + smart integration is compelling; Nest Protect is the gold standard

LOW-VALUE SMART HOME UPGRADES

1. Smart Refrigerators (ROI: Poor)
Cost: $2,000-$5,000
Value added: Minimal
Why: Buyers have their own appliance preferences. Technology becomes outdated before appliances wear out. Too niche.

2. Extensive Voice Assistant Integration (ROI: Poor)
Cost: Varies
Value added: Minimal
Why: Alexa/Google preferences vary. Easy for buyers to set up themselves. Not a differentiator.

3. Smart Kitchen Appliances (ROI: Poor)
Cost: High
Value added: Near zero
Why: Smart ovens, coffee makers, etc. are personal choices. Technology cycles faster than appliance replacement.

4. Overly Complex Automation (ROI: Negative)
Cost: High
Value added: Can actually hurt resale
Why: Complex systems intimidate average buyers. If they can't easily control it, it's a liability, not an asset. Keep it user-friendly.

BEST PRACTICES FOR VALUE-ADD TECH

1. Keep It Simple: If you need an engineering degree to explain it, it won't add value.

2. Choose Established Brands: Nest, Ring, Ecobee, August are recognized and trusted. Obscure brands raise concerns about support and longevity.

3. Ensure Interoperability: Systems should work with multiple platforms (Apple, Google, Alexa). Avoid proprietary lock-in.

4. Document Everything: Provide buyers with clear documentation, app access transfers, and setup instructions.

5. Focus on Energy Efficiency: Provable savings are compelling. Smart thermostats, LED lighting controls, and solar monitoring have quantifiable benefits.

6. Prioritize Security: Everyone cares about safety. Smart locks, cameras, and alarm systems have universal appeal.

WHAT BUYERS REALLY WANT

In surveys, homebuyers prioritize smart features in this order:
1. Security (cameras, smart locks, alarms)
2. Temperature control (smart thermostats)
3. Lighting control
4. Energy monitoring
5. Voice control (as part of other systems)

Notice: Complex entertainment systems and smart appliances don't make the list.

THE BOTTOM LINE

Smart home technology can add 3-5% to your home's value IF you choose the right upgrades. Focus on:
- Security
- Energy efficiency
- Convenience features with mainstream appeal
- User-friendly systems with good documentation

Skip the expensive gimmicks and invest in technology that makes everyday life easier. That's what buyers will pay for.

Use PropertyAI's valuation tool to see how your planned upgrades might impact your specific property's value."""
        }
    ]
    
    return jsonify({'blogs': blogs}), 200


# ============================================================================
# API ENDPOINTS - AI Chatbot (Gemini)
# ============================================================================

@app.route('/api/chat', methods=['POST'])
def chat_with_ai():
    """Chat with AI for real estate queries - with fallback responses"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').lower().strip()
        
        if not user_message:
            return jsonify({'response': 'Please enter a message.'}), 200
        
        # Try Gemini API first if available
        if CHATBOT_ENABLED and genai_client:
            try:
                system_context = """You are PropertyAI Assistant, an expert AI helper for real estate and property valuation. 
                Be helpful, concise, and professional. Keep responses under 150 words."""
                
                full_prompt = f"{system_context}\n\nUser: {user_message}\n\nAssistant:"
                
                response = genai_client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=full_prompt
                )
                
                if response and hasattr(response, 'text') and response.text:
                    return jsonify({'response': response.text, 'success': True}), 200
            except Exception as api_error:
                print(f"Gemini API Error: {api_error}")
        
        # Fallback: Pattern-based responses for common real estate queries
        responses = {
            'hello': "Hello! I'm your PropertyAI Assistant. I can help you with property valuations, EMI calculations, ROI analysis, and real estate investment tips. What would you like to know?",
            'hi': "Hi there! Welcome to PropertyAI. Ask me about property prices, market trends, mortgage calculations, or investment strategies!",
            'emi': "EMI (Equated Monthly Installment) is your monthly loan payment. Use our EMI Calculator in the Dashboard! Formula: EMI = P × r × (1+r)^n / ((1+r)^n-1), where P=principal, r=monthly rate, n=tenure in months.",
            'roi': "ROI (Return on Investment) measures your property investment returns. It includes rental income + capital appreciation. Use our ROI Calculator to project your returns over time!",
            'price': "Property prices depend on location, area, amenities, and market conditions. Use our Price Prediction feature on the Dashboard for AI-powered valuations based on your property details.",
            'invest': "Real estate investment tips: 1) Location is key, 2) Check rental yield (aim for 3-5%), 3) Consider appreciation potential, 4) Diversify your portfolio, 5) Always do due diligence.",
            'loan': "Home loan tips: 1) Compare interest rates across banks, 2) Keep EMI under 40% of income, 3) Longer tenure = lower EMI but more interest, 4) Consider prepayment options.",
            'market': "Real estate market trends vary by location. Generally, metro cities show steady appreciation (5-8% annually). Use our Analytics Dashboard for detailed market insights!",
            'buy': "Property buying checklist: 1) Verify legal documents, 2) Check builder reputation, 3) Inspect property physically, 4) Compare market prices, 5) Factor in hidden costs (registration, maintenance).",
            'rent': "Rental yield = (Annual Rent / Property Value) × 100. Good yield is 3-5%. Consider location demand, amenities, and maintenance costs when evaluating rental properties.",
            'help': "I can help with: 📊 Price predictions, 💰 EMI calculations, 📈 ROI analysis, 🏠 Market trends, 💡 Investment tips. Try our Dashboard features for detailed analysis!",
        }
        
        # Find matching response
        for keyword, resp in responses.items():
            if keyword in user_message:
                return jsonify({'response': resp}), 200
        
        # Default response
        return jsonify({
            'response': "I'm your PropertyAI Assistant! I can help with property valuations, EMI/ROI calculations, and investment advice. Try asking about: property prices, EMI, ROI, market trends, or investment tips. For predictions, use the Dashboard's prediction feature!"
        }), 200
        
    except Exception as e:
        print(f"Chatbot Error: {e}")
        return jsonify({'response': "I'm here to help with real estate questions. Try asking about property prices, EMI calculations, ROI, or investment tips!"}), 200


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    # Create templates and static directories if they don't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    print("\n" + "="*80)
    print(" "*25 + "FLASK APPLICATION")
    print("="*80)
    print("\n✓ Server starting...")
    print(f"✓ Database initialized")
    print(f"✓ Total users in system: {db.count_users()}")
    print("\n" + "="*80)
    print("Server running at: http://localhost:5000")
    print("="*80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
