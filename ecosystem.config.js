module.exports = {
  apps: [
    {
      name: "fastapi-server",
      script: "uvicorn",
      args: "main:app --host 0.0.0.0 --port 8000 --workers 1",
      interpreter: "python3",
      exec_mode: "fork",
      autorestart: true,
      watch: false,
      env: {
        PYTHONUNBUFFERED: "1",
      },
    },
  ],
};
