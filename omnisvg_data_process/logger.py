from loguru import logger

class Logger:
  def __init__(self, label: str):
    self.logger = logger
    self.label = label

  def info(self, message: str):
    self.logger.info(f"[{self.label}]: {message}")

  def warning(self, message: str):
    self.logger.warning(f"[{self.label}]: {message}")

  def error(self, message: str):
    self.logger.error(f"[{self.label}]: {message}")
    
  def debug(self, message: str):
    self.logger.debug(f"[{self.label}]: {message}")

  def critical(self, message: str):
    self.logger.critical(f"[{self.label}]: {message}")
    
    