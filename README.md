# Risk Manager Agent 🚀

**Production-level FastAPI microservice for DeFi portfolio risk monitoring**

The Risk Manager Agent is a comprehensive risk monitoring service that tracks user portfolios across major DeFi protocols in real-time to detect and prevent liquidation, over-exposure, and volatility risks.

## 🎯 Features

### Core Risk Monitoring
- **Real-time Health Ratio Calculation** - Monitor collateral-to-debt ratios across protocols
- **Liquidation Risk Detection** - Early warning system with configurable thresholds  
- **Volatility & VaR Analysis** - Value-at-Risk calculations using historical price data
- **Concentration Risk Scoring** - Portfolio diversification analysis
- **Cross-Protocol Analytics** - Unified view across Aave, Compound, Curve

### Data Sources
- **The Graph Subgraphs** - Real-time DeFi protocol data (Aave V2/V3, Compound)
- **CoinGecko API** - Historical price data and market metrics
- **Alchemy RPC** - On-chain balance and transaction data
- **Etherscan API** - Gas price monitoring and network status
- **DefiLlama API** - Protocol TVL and liquidity metrics

### Advanced Features
- **Background Processing** - Automated risk monitoring every 10 minutes
- **Smart Alert System** - Contextual alerts with actionable recommendations
- **Machine Learning** - Optional XGBoost model for liquidation probability prediction
- **Rate Limiting** - Redis-based throttling (60 req/min per wallet)
- **Structured Logging** - Integration with LokiAI logging infrastructure

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI API   │────│  Risk Engine     │────│  External APIs  │
│   (Port 8001)   │    │  (Core Logic)    │    │  (Data Sources) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         │              ┌──────────────────┐               │
         │              │  Background      │               │
         └──────────────│  Tasks Runner    │───────────────┘
                        └──────────────────┘
                                 │
                    ┌─────────────────────────────┐
                    │     Data Storage Layer      │
                    │                             │
                    │  ┌─────────────┐ ┌─────────┐│
                    │  │  MongoDB    │ │  Redis  ││
                    │  │  (Atlas)    │ │ (Cache) ││
                    │  └─────────────┘ └─────────┘│
                    └─────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- MongoDB Atlas account
- API keys for external services
- Redis (optional, for production)

### Installation

1. **Clone and Install Dependencies**
```bash
git clone <repository_url>
cd "Risk Manager"
pip install -r requirements.txt
```

2. **Configure Environment**
```bash
cp .env.example .env
# Edit .env with your API keys and database URLs
```

3. **Start the Service**
```bash
python run.py
```

The service will be available at `http://localhost:8001`

### Using Docker (Recommended for Production)
```bash
docker build -t risk-manager-agent .
docker run -p 8001:8001 --env-file .env risk-manager-agent
```

## 📊 API Endpoints

### System Status
```http
GET /api/risk/status
```
Returns overall system health, tracked wallets, and performance metrics.

### Risk Summary
```http
GET /api/risk/summary?wallet={address}
Headers: x-wallet-address: {address}
```
Comprehensive risk analysis for a specific wallet across all protocols.

### Manual Analysis  
```http
POST /api/risk/analyze
Headers: x-wallet-address: {address}
Content-Type: application/json

{
  "wallet_address": "0x...",
  "force_refresh": true,
  "protocols": ["aave_v3", "compound"]
}
```

### Risk Alerts
```http  
GET /api/risk/alerts?severity=high&resolved=false
Headers: x-wallet-address: {address}
```
Retrieve risk alerts with filtering options.

### Resolve Alert
```http
POST /api/risk/alerts/{alert_id}/resolve  
Headers: x-wallet-address: {address}
```

## 🔧 Configuration

### Environment Variables

```bash
# Database Configuration
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/
MONGO_DB_NAME=loki_agents
REDIS_URL=redis://localhost:6379/0
ENABLE_REDIS=true

# API Keys
COINGECKO_API_KEY=your_key_here
ALCHEMY_API_KEY=your_key_here  
ETHERSCAN_API_KEY=your_key_here
THEGRAPH_API_KEY=your_key_here

# Service Configuration
RISK_MANAGER_PORT=8001
RATE_LIMIT_PER_MINUTE=60
BACKGROUND_TASK_INTERVAL=600
LOKI_URL=http://localhost:8000

# Risk Parameters
LIQUIDATION_THRESHOLD=1.0
SAFETY_THRESHOLD=1.2
VAR_CONFIDENCE=0.95
VOLATILITY_WINDOW_DAYS=7

# ML Configuration  
MODEL_RETRAIN_HOURS=24
MIN_TRAINING_SAMPLES=100
```

### Risk Thresholds

The service uses configurable thresholds for different risk levels:

- **Liquidation Warning**: Health ratio < 1.2
- **Liquidation Critical**: Health ratio < 1.1  
- **High Volatility**: Daily volatility > 5%
- **Concentration Warning**: >50% portfolio in single asset
- **Low Liquidity**: <$100K daily trading volume

## 🧮 Risk Calculation Details

### Health Ratio Calculation
```python
health_ratio = collateral_value / (borrow_value * safety_threshold)
```

### Liquidation Risk Score
```python
if health_ratio > 2.0:
    risk_score = 0
elif health_ratio > 1.5:  
    risk_score = 20 * (2.0 - health_ratio) / 0.5
elif health_ratio > 1.0:
    risk_score = 60 + 35 * (1.2 - health_ratio) / 0.2  
else:
    risk_score = 95 + 5 * max(0, 1.0 - health_ratio)
```

### Value at Risk (VaR)
Using historical price data to calculate 95% VaR:
```python
var_95 = np.percentile(returns, 5)  # 5th percentile for 95% VaR
```

### Exposure Score
Combined risk metric weighing multiple factors:
```python
exposure_score = (
    0.3 * concentration_score +
    0.3 * volatility_component + 
    0.2 * liquidity_component +
    0.2 * correlation_risk
)
```

## 🤖 Machine Learning

### XGBoost Liquidation Predictor

The optional ML module trains an XGBoost classifier to predict liquidation probability:

**Features Used:**
- Health ratio
- Total supplied/borrowed USD
- Volatility score  
- Liquidity score
- Concentration score
- Gas price (gwei)
- Market fear/greed index
- BTC/ETH price changes

**Model Performance Tracking:**
- Accuracy, Precision, Recall
- ROC-AUC score
- Feature importance analysis
- Automatic retraining every 24h

### Usage
```python
# Get predictions
prob_24h, prob_7d = await get_liquidation_predictions(features)

# Enhance risk summary with ML
enhanced_summary = await enhance_risk_with_ml(risk_summary)
```

## 📊 Monitoring & Observability

### Structured Logging
All events are logged with structured data:
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "service": "risk_manager", 
  "event_type": "wallet_risk_monitored",
  "level": "info",
  "data": {
    "wallet_address": "0x...",
    "risk_score": 25.4,
    "alerts_generated": 1
  }
}
```

### Metrics Collection
- Request count and response times
- API health status monitoring  
- Database connection health
- Background task execution stats
- Risk distribution analytics

### Integration with LokiAI
Automatic logging to the LokiAI ecosystem:
```python
await log_to_loki("critical_alert", {
    "wallet_address": "0x...",
    "risk_type": "liquidation",
    "severity": "critical"
})
```

## 🔒 Security Features

### Authentication & Authorization
- Required `x-wallet-address` header on all routes
- Wallet address format validation
- Request/wallet address matching verification

### Rate Limiting
- Redis-based sliding window rate limiting
- 60 requests per minute per wallet address
- IP-based secondary limits
- Graceful degradation when Redis unavailable

### Input Validation & Sanitization
- Pydantic model validation
- SQL injection prevention
- XSS protection on string inputs
- Request size limiting

### Security Monitoring
- Suspicious activity detection
- Multiple IP access pattern analysis  
- High-frequency request alerting
- Security event logging

## 🏃‍♂️ Performance Optimization

### Async Architecture
- Full async/await implementation
- Concurrent API requests using `httpx`
- Non-blocking database operations with `motor`
- Background task processing

### Caching Strategy
- Redis caching for frequently accessed data
- API response caching with TTL
- Database query result caching
- Rate limit data caching

### Database Optimization
- Strategic MongoDB indexing
- Query result pagination
- Connection pooling
- Bulk operations for batch updates

### API Efficiency  
- Retry logic with exponential backoff
- Connection reuse and pooling
- Request batching where possible
- Selective data fetching

## 🧪 Testing

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
```

### Test Categories
- **Unit Tests**: Individual component testing
- **Integration Tests**: API endpoint testing  
- **Performance Tests**: Load and stress testing
- **Security Tests**: Authentication and input validation

## 🚀 Deployment

### Production Checklist
- [ ] Set `ENV=production` in environment
- [ ] Configure MongoDB Atlas with authentication
- [ ] Set up Redis cluster for caching
- [ ] Configure proper firewall rules
- [ ] Set up log aggregation
- [ ] Configure monitoring and alerting
- [ ] Set up SSL/TLS termination
- [ ] Configure backup strategy

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8001

CMD ["python", "run.py", "--env", "production"]
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment  
metadata:
  name: risk-manager-agent
spec:
  replicas: 3
  selector:
    matchLabels:
      app: risk-manager-agent
  template:
    metadata:
      labels:
        app: risk-manager-agent
    spec:
      containers:
      - name: risk-manager
        image: risk-manager-agent:latest
        ports:
        - containerPort: 8001
        env:
        - name: MONGODB_URI
          valueFrom:
            secretKeyRef:
              name: risk-manager-secrets
              key: mongodb-uri
```

## 📈 Scaling Considerations

### Horizontal Scaling
- Stateless service design enables multiple instances
- Redis for shared state and rate limiting
- MongoDB connection pooling across instances
- Load balancer configuration

### Vertical Scaling
- Background task worker pool sizing
- Database connection pool tuning
- Memory optimization for large datasets
- CPU optimization for ML workloads

### Performance Monitoring
- Response time percentiles (p50, p95, p99)
- Throughput monitoring (requests per second)
- Error rate tracking
- Resource utilization monitoring

## 🐛 Troubleshooting

### Common Issues

**MongoDB Connection Errors**
```bash
# Check connection string format
# Verify network access to Atlas cluster  
# Confirm authentication credentials
```

**API Key Issues**
```bash
# Verify all required API keys are set
# Check API key validity and rate limits
# Test individual API endpoints
```

**High Memory Usage**
```bash
# Check background task frequency
# Monitor ML model memory usage
# Review database query efficiency  
```

**Rate Limiting Issues**
```bash
# Verify Redis connection
# Check rate limit configuration
# Monitor request patterns
```

### Debug Mode
```bash
python run.py --log-level DEBUG --reload
```

## 🤝 Contributing

### Development Setup
```bash
git clone <repository>
cd "Risk Manager"
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Code Quality
- Black code formatting
- isort import sorting
- flake8 linting  
- mypy type checking
- Pre-commit hooks

## 📄 License

Proprietary - LokiAI Development Team

## 🆘 Support

For technical support and questions:
- Email: dev@lokiai.com
- Documentation: [Internal Wiki]
- Issue Tracking: [Internal JIRA]

---

**Risk Manager Agent v1.0.0** - Built with ❤️ for the LokiAI ecosystem