from flask import Flask, jsonify, request
import datetime
import uuid

# Create Flask application
app = Flask(__name__)

# Simple in-memory storage
job_offers = []
applications = []

@app.route('/')
def index():
    return jsonify({'status': 'API is running', 'version': '1.0.0'})

@app.route('/api/job-offers', methods=['GET'])
def get_job_offers():
    return jsonify(job_offers)

@app.route('/api/job-offers', methods=['POST'])
def create_job_offer():
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400
        
    data = request.get_json()
    
    # Set default values for optional fields
    job_offer = {
        'id': str(uuid.uuid4()),
        'title': data.get('title', 'Untitled Position'),
        'company': data.get('company', 'Unnamed Company'),
        'location': data.get('location', 'Not Specified'),
        'description': data.get('description', ''),
        'keywords': data.get('keywords', []),
        'requirements': data.get('requirements', []),
        'created_at': datetime.datetime.now().isoformat(),
        'status': data.get('status', 'active')
    }
    
    job_offers.append(job_offer)
    
    return jsonify({
        'message': 'Job offer created successfully',
        'offer_id': job_offer['id']
    }), 201

@app.route('/api/job-offers/<offer_id>', methods=['GET'])
def get_job_offer(offer_id):
    # Find job offer by ID
    for offer in job_offers:
        if offer['id'] == offer_id:
            return jsonify(offer)
            
    return jsonify({'error': 'Job offer not found'}), 404

@app.route('/api/applications', methods=['POST'])
def create_application():
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400
        
    data = request.get_json()
    
    # Validate required fields
    if 'offer_id' not in data or 'cv_text' not in data:
        return jsonify({'error': 'offer_id and cv_text are required'}), 400
        
    # Check if job offer exists
    offer_exists = False
    for offer in job_offers:
        if offer['id'] == data['offer_id']:
            offer_exists = True
            break
            
    if not offer_exists:
        return jsonify({'error': 'Job offer not found'}), 404
        
    # Create application
    application = {
        'id': str(uuid.uuid4()),
        'offer_id': data['offer_id'],
        'name': data.get('name', 'Anonymous Candidate'),
        'email': data.get('email', ''),
        'phone': data.get('phone', ''),
        'cv_text': data['cv_text'],
        'skills': data.get('skills', []),
        'match_score': data.get('match_score', 50.0),
        'status': 'pending',
        'created_at': datetime.datetime.now().isoformat()
    }
    
    applications.append(application)
    
    return jsonify({
        'message': 'Application submitted successfully',
        'application_id': application['id']
    }), 201

@app.route('/api/applications/<application_id>', methods=['GET'])
def get_application(application_id):
    # Find application by ID
    for application in applications:
        if application['id'] == application_id:
            return jsonify(application)
            
    return jsonify({'error': 'Application not found'}), 404

if __name__ == '__main__':
    # Add some sample data
    job_offers.append({
        'id': 'sample-job-1',
        'title': 'Senior Software Engineer',
        'company': 'Tech Solutions Inc.',
        'location': 'Remote',
        'description': 'We are looking for a skilled Senior Software Engineer with expertise in Python and web frameworks.',
        'keywords': ['python', 'flask', 'django', 'api', 'web development'],
        'requirements': ['5+ years of experience', 'Python expertise', 'Web development experience'],
        'created_at': datetime.datetime.now().isoformat(),
        'status': 'active'
    })
    
    print("Simple Job API starting - visit http://localhost:5000/api/job-offers to see sample data")
    app.run(debug=True, host='0.0.0.0', port=5000) 