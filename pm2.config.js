module.exports = {
  apps: [
    {
      name: "SLAI-CollabRouter",
      script: "main_collaborative.py",
      interpreter: "python3",
      watch: false,
      autorestart: true,
      max_memory_restart: "2G",
      env: {
        NODE_ENV: "production"
      },
      out_file: "./logs/collab_out.log",
      error_file: "./logs/collab_err.log",
      merge_logs: true
    },
    {
      name: "SLAI-Frontend",
      script: "frontend.py",
      interpreter: "python3",
      watch: false,
      autorestart: true,
      max_memory_restart: "2G",
      env: {
        NODE_ENV: "production"
      },
      out_file: "./logs/frontend_out.log",
      error_file: "./logs/frontend_err.log",
      merge_logs: true
    },
    {
      name: "SLAI-AgentRunner",
      script: "agent_runner.py",
      interpreter: "python3",
      watch: false,
      autorestart: true,
      max_memory_restart: "2G",
      out_file: "./logs/agent_out.log",
      error_file: "./logs/agent_err.log",
      merge_logs: true
    }
  ]
};
