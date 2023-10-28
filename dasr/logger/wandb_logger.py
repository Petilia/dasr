import wandb
from datetime import datetime

class WandbLogger:
    def __init__(self, project_name, run_name):
        wandb.login()
        wandb.init(project=project_name, 
                   name=run_name)
        self.step = 0
        self.mode = ""
        
    def log_dict(self, stats):
        for key, value in stats.items():
            if key == "epoch":
                log_name = key
            else:
                log_name = f"{self.mode}/{key}"

            wandb.log({
                log_name: value,
            }, step=self.step)
            
    def log_best_model(self, path):
        wandb.save(path)
            
    def set_mode(self, mode):
        self.mode = mode
        
    def set_step(self, step):
        self.step = step
        
    