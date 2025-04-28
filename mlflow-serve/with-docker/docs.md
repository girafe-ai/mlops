```
┌────────────────────────────────────┐
│            User (Browser)          │
│       (localhost:9000, Frontend)   │
└────────────────────────────────────┘
                 │
                 ▼
      Uploads Image (POST /invocations)
                 │
                 ▼
┌────────────────────────────────────-┐
│        Proxy Server (Flask)         │
│         (localhost:8890)            │
│    - Handles CORS                   │
│    - Forwards to MLflow             │
└────────────────────────────────────-┘
                 │
                 ▼
  Forwards JSON data (POST /invocations)
                 │
                 ▼
┌────────────────────────────────────-┐
│          MLflow Model Server        │
│          (localhost:8888)           │
│    - Loads model artifacts          │
│    - Runs inference (cat / dog)     │
└────────────────────────────────────-┘
                 │
                 ▼
       Returns Prediction (🐶 / 🐱)
                 │
                 ▼
┌────────────────────────────────────-┐
│        Proxy Server (Flask)         │
└────────────────────────────────────-┘
                 │
                 ▼
┌────────────────────────────────────-┐
│            User (Browser)           │
│         Shows prediction result     │
└────────────────────────────────────-┘

```