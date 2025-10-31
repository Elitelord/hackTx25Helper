# Install build deps for some Python packages (kept minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements to leverage Docker cache
COPY requirements.txt .

# Install into a wheelhouse (local cache) then install from there to speed subsequent builds
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip wheel --no-cache-dir --wheel-dir=/wheels -r requirements.txt

# ---- Final runtime image ----
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System deps needed at runtime (kept minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy wheels from builder and install them (faster & reproducible)
COPY --from=builder /wheels /wheels
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11

# Install from wheels
RUN pip install --no-index --find-links=/wheels -r /wheels/../requirements.txt || true

# Fallback: if above fails (some packages), install from PyPI as last resort
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port used by uvicorn
EXPOSE 8000

# Use environment variables to configure number of workers / host / port if desired
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--loop", "asyncio"]