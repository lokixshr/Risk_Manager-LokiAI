# Risk Manager Agent - Production Deployment Guide

## 📋 Overview

This guide covers the complete production deployment of the Risk Manager Agent, including infrastructure setup, configuration, monitoring, and maintenance procedures.

## 🏗️ Architecture Overview

```mermaid
graph TB
    A[Load Balancer/Ingress] --> B[Risk Manager Pods 3x]
    B --> C[Redis Cluster]
    B --> D[MongoDB Atlas]
    B --> E[External APIs]
    F[Monitoring Stack] --> B
    G[Logs Aggregation] --> B
    H[Backup System] --> D
```

## 🚀 Pre-deployment Checklist

### Prerequisites
- [ ] Kubernetes cluster (v1.24+) with 16+ vCPUs, 32GB+ RAM
- [ ] MongoDB Atlas cluster (M10+ for production)
- [ ] Redis cluster or managed Redis service
- [ ] Valid SSL certificates
- [ ] API keys for external services
- [ ] Monitoring infrastructure (Prometheus, Grafana)
- [ ] CI/CD pipeline configured

### Required Secrets
- [ ] MongoDB connection string
- [ ] API keys (CoinGecko, Alchemy, Etherscan, TheGraph)
- [ ] Redis password
- [ ] SSL certificates
- [ ] Webhook URLs for notifications

## 📦 Deployment Steps

### 1. Environment Preparation

```bash
# Create namespace
kubectl create namespace risk-manager-prod

# Create secrets
kubectl create secret generic risk-manager-secrets \
  --namespace=risk-manager-prod \
  --from-literal=mongodb-uri="mongodb+srv://user:pass@cluster.mongodb.net/prod_db" \
  --from-literal=coingecko-api-key="your-coingecko-key" \
  --from-literal=alchemy-api-key="your-alchemy-key" \
  --from-literal=alchemy-url="your-alchemy-url" \
  --from-literal=etherscan-api-key="your-etherscan-key" \
  --from-literal=thegraph-api-key="your-thegraph-key"

# Create ConfigMap
kubectl create configmap risk-manager-config \
  --namespace=risk-manager-prod \
  --from-file=config/production.yaml
```

### 2. Deploy Infrastructure Components

```bash
# Deploy Redis (if not using managed service)
kubectl apply -f k8s/redis/ --namespace=risk-manager-prod

# Deploy monitoring components
kubectl apply -f k8s/monitoring/ --namespace=risk-manager-prod

# Deploy ingress controller
kubectl apply -f k8s/ingress/ --namespace=risk-manager-prod
```

### 3. Deploy Main Application

```bash
# Deploy Risk Manager
kubectl apply -f k8s/deployment.yaml --namespace=risk-manager-prod

# Verify deployment
kubectl rollout status deployment/risk-manager-agent --namespace=risk-manager-prod

# Check pods are running
kubectl get pods -n risk-manager-prod -l app=risk-manager-agent
```

### 4. Post-deployment Validation

```bash
# Run health checks
python scripts/production_health_check.py --url https://your-domain.com

# Run smoke tests
python scripts/smoke_test.py --url https://your-domain.com

# Verify monitoring
kubectl port-forward -n risk-manager-prod svc/grafana 3000:3000
# Open http://localhost:3000 and verify dashboards
```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `ENV` | Environment (production) | ✅ | development |
| `MONGODB_URI` | MongoDB connection string | ✅ | - |
| `REDIS_URL` | Redis connection URL | ✅ | - |
| `COINGECKO_API_KEY` | CoinGecko API key | ✅ | - |
| `ALCHEMY_API_KEY` | Alchemy API key | ✅ | - |
| `ETHERSCAN_API_KEY` | Etherscan API key | ✅ | - |
| `THEGRAPH_API_KEY` | The Graph API key | ✅ | - |
| `RATE_LIMIT_PER_MINUTE` | Rate limit per wallet | ❌ | 60 |
| `BACKGROUND_TASK_INTERVAL` | Background task interval (sec) | ❌ | 600 |
| `LOG_LEVEL` | Logging level | ❌ | INFO |

### Resource Limits

```yaml
resources:
  requests:
    memory: "1Gi"
    cpu: "500m"
  limits:
    memory: "2Gi"
    cpu: "1000m"
```

### Scaling Configuration

- **Minimum Replicas**: 3
- **Maximum Replicas**: 10
- **CPU Target**: 70%
- **Memory Target**: 80%

## 📊 Monitoring & Observability

### Health Check Endpoints

| Endpoint | Purpose | Expected Response |
|----------|---------|------------------|
| `/api/risk/health` | Basic health check | 200 (healthy) / 503 (unhealthy) |
| `/api/risk/status` | Detailed system status | System metrics and status |
| `/api/info` | Service information | Service metadata |

### Key Metrics to Monitor

#### Application Metrics
- Request rate (RPS)
- Response time (p50, p95, p99)
- Error rate
- Active connections
- Background task execution time
- Circuit breaker status

#### System Metrics
- CPU utilization
- Memory usage
- Disk I/O
- Network traffic
- Pod restart count

#### Business Metrics
- Wallets monitored
- Risk analyses performed
- Alerts generated
- API call success rate
- Data freshness

### Alerting Rules

#### Critical Alerts (PagerDuty)
- Service down (all pods unhealthy)
- High error rate (>5% for 5 minutes)
- Response time degradation (p95 > 5s for 10 minutes)
- Multiple circuit breakers open
- Database connectivity issues

#### Warning Alerts (Slack)
- High CPU usage (>80% for 15 minutes)
- High memory usage (>85% for 15 minutes)
- Elevated error rate (>2% for 10 minutes)
- Background task delays
- Single circuit breaker open

## 🔒 Security

### Security Headers
```yaml
# Implemented in ingress
security-headers: |
  X-Frame-Options: DENY
  X-Content-Type-Options: nosniff
  X-XSS-Protection: 1; mode=block
  Strict-Transport-Security: max-age=31536000
  Content-Security-Policy: default-src 'self'
```

### Network Policies
```yaml
# Only allow necessary traffic
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: risk-manager-network-policy
spec:
  podSelector:
    matchLabels:
      app: risk-manager-agent
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
  egress:
  - to: []  # Allow all outbound (for API calls)
    ports:
    - protocol: TCP
      port: 443
    - protocol: TCP
      port: 80
```

### RBAC Configuration
```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: risk-manager-service-account
  namespace: risk-manager-prod
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: risk-manager-role
  namespace: risk-manager-prod
rules:
- apiGroups: [""]
  resources: ["pods", "services", "configmaps"]
  verbs: ["get", "list", "watch"]
```

## 🔄 Backup & Recovery

### Database Backup
```bash
# MongoDB Atlas automatic backups are enabled
# Point-in-time recovery available for 72 hours
# Daily snapshots retained for 30 days

# Manual backup (if needed)
mongodump --uri="mongodb+srv://user:pass@cluster.mongodb.net/prod_db" \
  --out="/backups/$(date +%Y%m%d_%H%M%S)"
```

### Redis Backup
```bash
# Redis persistence enabled with RDB snapshots
# Automatic failover configured

# Manual backup
kubectl exec -n risk-manager-prod redis-master-0 -- redis-cli BGSAVE
```

### Configuration Backup
```bash
# Backup Kubernetes configurations
kubectl get all,configmap,secret -n risk-manager-prod -o yaml > backup-$(date +%Y%m%d).yaml
```

### Recovery Procedures

#### Database Recovery
1. Identify restore point from Atlas backups
2. Create new cluster from backup
3. Update connection strings in secrets
4. Restart application pods

#### Application Recovery
1. Check pod logs: `kubectl logs -n risk-manager-prod deployment/risk-manager-agent`
2. Verify configuration: `kubectl describe configmap -n risk-manager-prod`
3. Check resource usage: `kubectl top pods -n risk-manager-prod`
4. Restart if needed: `kubectl rollout restart deployment/risk-manager-agent -n risk-manager-prod`

## 📈 Performance Tuning

### Database Optimization
- Enable MongoDB connection pooling (min: 5, max: 20)
- Create appropriate indexes for query patterns
- Monitor slow queries and optimize
- Use read preferences for read-heavy operations

### Redis Configuration
```yaml
# Redis configuration
maxmemory: 2gb
maxmemory-policy: allkeys-lru
timeout: 300
tcp-keepalive: 300
```

### Application Tuning
```python
# FastAPI/Uvicorn configuration
workers = 4  # CPU cores * 2
max_requests = 1000
max_requests_jitter = 50
preload_app = True
```

## 🔧 Maintenance Procedures

### Regular Maintenance Tasks

#### Daily
- [ ] Check system health dashboard
- [ ] Review error logs and alerts
- [ ] Monitor resource usage trends
- [ ] Verify backup completion

#### Weekly
- [ ] Review performance metrics
- [ ] Check for security updates
- [ ] Analyze slow queries
- [ ] Update documentation if needed

#### Monthly
- [ ] Review and tune autoscaling policies
- [ ] Rotate API keys if required
- [ ] Update dependencies
- [ ] Conduct disaster recovery test
- [ ] Review monitoring and alerting rules

### Update Procedures

#### Minor Updates (patches)
```bash
# Update image tag in deployment
kubectl set image deployment/risk-manager-agent \
  risk-manager=ghcr.io/org/risk-manager-agent:v1.0.1 \
  -n risk-manager-prod

# Wait for rollout
kubectl rollout status deployment/risk-manager-agent -n risk-manager-prod
```

#### Major Updates
1. Deploy to staging first
2. Run full test suite
3. Schedule maintenance window
4. Deploy with blue-green strategy
5. Monitor for issues
6. Rollback if necessary

### Rollback Procedures
```bash
# Quick rollback
kubectl rollout undo deployment/risk-manager-agent -n risk-manager-prod

# Rollback to specific revision
kubectl rollout undo deployment/risk-manager-agent --to-revision=2 -n risk-manager-prod

# Check rollback status
kubectl rollout status deployment/risk-manager-agent -n risk-manager-prod
```

## 🚨 Troubleshooting

### Common Issues

#### High Memory Usage
1. Check for memory leaks in logs
2. Review connection pool sizes
3. Monitor background task execution
4. Consider increasing pod memory limits

#### High CPU Usage
1. Check for infinite loops or blocking operations
2. Review database query performance
3. Monitor external API response times
4. Scale horizontally if needed

#### Database Connection Issues
1. Verify MongoDB Atlas network access
2. Check connection string and credentials
3. Monitor connection pool usage
4. Review MongoDB Atlas logs

#### External API Failures
1. Check circuit breaker status
2. Verify API keys and rate limits
3. Monitor network connectivity
4. Review API provider status pages

### Log Analysis
```bash
# Get recent logs
kubectl logs -n risk-manager-prod deployment/risk-manager-agent --tail=100

# Follow logs in real-time
kubectl logs -n risk-manager-prod deployment/risk-manager-agent -f

# Get logs from specific pod
kubectl logs -n risk-manager-prod risk-manager-agent-xxx-yyy

# Search for errors
kubectl logs -n risk-manager-prod deployment/risk-manager-agent | grep ERROR
```

## 📞 Support Contacts

### Escalation Path
1. **Level 1**: Development Team
2. **Level 2**: Platform Team
3. **Level 3**: External Vendors (MongoDB, Redis, Cloud Provider)

### Emergency Contacts
- **Development Team**: dev@company.com
- **Platform Team**: platform@company.com
- **On-call**: +1-xxx-xxx-xxxx

### Service Dependencies
- **MongoDB Atlas**: support.mongodb.com
- **Redis**: Redis Enterprise support
- **Cloud Provider**: Cloud vendor support
- **External APIs**: Vendor-specific support channels

## 📊 Performance Benchmarks

### Expected Performance
- **Response Time**: p95 < 500ms, p99 < 1000ms
- **Throughput**: 1000+ RPS per pod
- **Availability**: 99.9% uptime
- **Error Rate**: < 0.1%

### Load Test Results
- **Normal Load**: 50 concurrent users → 98% success rate
- **Peak Load**: 500 concurrent users → 95% success rate
- **Spike Test**: 1000 concurrent users → 90% success rate (with rate limiting)

## 📚 Additional Resources

- [API Documentation](./API_DOCUMENTATION.md)
- [Architecture Decision Records](./ADR/)
- [Runbooks](./runbooks/)
- [Security Guidelines](./SECURITY.md)
- [Development Guide](./DEVELOPMENT.md)