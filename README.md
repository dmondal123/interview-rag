# Interactive Programming Quiz Application

An AI-powered quiz application that adaptively tests users on SQL and Java programming concepts, providing real-time feedback and personalized difficulty adjustments.

## 🎯 Features

### Core Functionality
- **Adaptive Difficulty**: Questions automatically adjust based on user performance
- **Multi-Domain Support**: Covers both SQL and Java programming concepts
- **Real-Time Feedback**: Immediate response validation and scoring
- **Performance Analytics**: Detailed feedback report upon quiz completion
- **Smart Question Generation**: AI-powered question variations to test concept understanding

### Technical Features
- **FAISS-HNSW Integration**: Efficient similarity search for question retrieval
- **OpenAI Integration**: Leverages GPT-3.5-turbo for answer evaluation and feedback
- **Distributed Caching**: Redis-based question pool management
- **Asynchronous Processing**: Parallel question loading and evaluation
- **Progress Tracking**: Real-time progress indicators during quiz sessions

## 🏗️ Architecture

### Components
1. **Frontend Interface**
   - Built with Chainlit for interactive chat experience
   - Real-time progress updates
   - Dynamic question rendering

2. **Question Management**
   - FAISS-HNSW vector database for question storage
   - Cached question pools with Redis
   - Difficulty-based question categorization

3. **Answer Evaluation**
   - Multi-layer validation system
   - Domain-specific answer checking
   - Confidence scoring mechanism

4. **Performance Analytics**
   - User performance tracking
   - Domain-specific metrics
   - Difficulty progression analysis

## 🚀 Implementation Details

### Question Pool Structure

'''python
{
"topic": {
"easy": [...questions...],
"medium": [...questions...],
"hard": [...questions...]
}
}
'''

### Difficulty Progression
- Easy → Medium → Hard (on correct answers)
- Hard → Medium → Easy (on incorrect answers)

### Scoring System
- Points awarded based on difficulty level
- Partial credit for partially correct answers
- Confidence-based scoring adjustments

## 📊 Data Management

### Storage
- PostgreSQL for persistent question storage
- Redis for distributed caching
- In-memory state management for active sessions

### Privacy & Security
- Anonymized user data
- 30-day data retention policy
- Encrypted storage for sensitive information

## 🔧 Technical Requirements

'''bash
pip install -r requirements.txt
'''

### Environment Variables

'''bash
OPENAI_API_KEY=your_openai_api_key
PG_DATABASE=your_database_name
PG_HOST=5432
PG_USER=your_username
PG_PASSWORD=your_password
'''

## 🎮 Usage

### Starting the Application

'''bash
chainlit run frontend.py
'''

### Quiz Flow
1. User starts quiz session
2. System loads question pool (with progress indicator)
3. Questions presented with adaptive difficulty
4. Real-time feedback on answers
5. Comprehensive performance report at completion


