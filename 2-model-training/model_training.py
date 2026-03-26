import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from models.ssd_cnn_256 import create_model, NUM_CLASSES, CLASS_NAMES
from models.losses import create_ssd_loss
from models.dataloader import create_dataloaders


# ============================================================================
# CONFIGURATION
# ============================================================================
DEFAULT_CONFIG = {
    'data_root': '../../data/processed',
    'output_dir': '../models_fine_tuned',
    'log_dir': '../logs',
    
    # Training params
    'epochs': 30,              # Augmenté de 50 → 30 (suffisant vu la convergence)
    'batch_size': 16,
    'learning_rate': 5e-4,     #  1e-3 → 5e-4 (fine-tuning)
    'weight_decay': 1e-4,
    
    # Loss params
    'focal_alpha': 0.25,
    'focal_gamma': 2.0,
    'reg_weight': 1.0,
    
    # Scheduler params
    'lr_patience': 3,          #  de 5 → 3 (réagir plus vite)
    'lr_factor': 0.5,
    
    # Early stopping
    'early_stop_patience': 8,  #  de 10 → 8
    
    # Device
    'device': 'cpu',  #  CPU only
    'num_workers': 0
}


# ============================================================================
# TRAINER CLASS
# ============================================================================
class Trainer:
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device(config['device'])
        
        # Create directories
        Path(config['output_dir']).mkdir(parents=True, exist_ok=True)
        Path(config['log_dir']).mkdir(parents=True, exist_ok=True)
        
        # Timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("=" * 60)
        print("🚀 XRAYVISION - SSD-CNN-256 Training")
        print("=" * 60)
        print(f"   Device: {self.device}")
        print(f"   Epochs: {config['epochs']}")
        print(f"   Batch size: {config['batch_size']}")
        print(f"   Learning rate: {config['learning_rate']}")
        print("=" * 60)
        
        # Initialize components
        self._init_model()
        self._init_data()
        self._init_training()
        
        # Tracking
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.train_losses = []
        self.val_losses = []
    
    def _init_model(self):
        """Initialize model"""
        print("\n📦 Initializing model...")
        self.model = create_model(num_classes=NUM_CLASSES)
        self.model.to(self.device)
    
    def _init_data(self):
        """Initialize dataloaders"""
        print("\n📂 Loading data...")
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            data_root=self.config['data_root'],
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers']
        )
    
    def _init_training(self):
        """Initialize optimizer, scheduler, loss"""
        print("\n⚙️ Initializing training components...")
        
        # Loss
        self.criterion = create_ssd_loss(
            num_classes=NUM_CLASSES,
            alpha=self.config['focal_alpha'],
            gamma=self.config['focal_gamma'],
            reg_weight=self.config['reg_weight']
        )
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.config['lr_factor'],
            patience=self.config['lr_patience']
        )
        
        print("   ✅ Loss: Focal Loss + Smooth L1")
        print("   ✅ Optimizer: Adam")
        print("   ✅ Scheduler: ReduceLROnPlateau")
    
    def train_epoch(self) -> dict:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_cls_loss = 0.0
        total_reg_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, cls_targets, reg_targets, pos_mask) in enumerate(self.train_loader):
            # Move to device
            images = images.to(self.device)
            cls_targets = cls_targets.to(self.device)
            reg_targets = reg_targets.to(self.device)
            pos_mask = pos_mask.to(self.device)
            
            # Forward pass
            cls_preds, reg_preds = self.model(images)
            
            # Compute loss
            loss, cls_loss, reg_loss = self.criterion(
                cls_preds, reg_preds, cls_targets, reg_targets, pos_mask
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            
            self.optimizer.step()
            
            # Accumulate
            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_reg_loss += reg_loss.item()
            num_batches += 1
            
            # Progress
            if (batch_idx + 1) % 10 == 0:
                print(f"      Batch {batch_idx + 1}/{len(self.train_loader)}: "
                      f"Loss={loss.item():.4f} (cls={cls_loss.item():.4f}, reg={reg_loss.item():.4f})")
        
        return {
            'loss': total_loss / num_batches,
            'cls_loss': total_cls_loss / num_batches,
            'reg_loss': total_reg_loss / num_batches
        }
    
    @torch.no_grad()
    def validate(self) -> dict:
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0.0
        total_cls_loss = 0.0
        total_reg_loss = 0.0
        num_batches = 0
        
        for images, cls_targets, reg_targets, pos_mask in self.val_loader:
            images = images.to(self.device)
            cls_targets = cls_targets.to(self.device)
            reg_targets = reg_targets.to(self.device)
            pos_mask = pos_mask.to(self.device)
            
            cls_preds, reg_preds = self.model(images)
            
            loss, cls_loss, reg_loss = self.criterion(
                cls_preds, reg_preds, cls_targets, reg_targets, pos_mask
            )
            
            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_reg_loss += reg_loss.item()
            num_batches += 1
        
        return {
            'loss': total_loss / max(num_batches, 1),
            'cls_loss': total_cls_loss / max(num_batches, 1),
            'reg_loss': total_reg_loss / max(num_batches, 1)
        }
    
    def save_checkpoint(self, path: str, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'num_classes': NUM_CLASSES,
            'class_names': CLASS_NAMES
        }
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = str(Path(path).parent / "best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"   💾 Saved best model to {best_path}")
    
    def train(self):
        """Main training loop"""
        print("\n" + "=" * 60)
        print("🏋️ Starting training...")
        print("=" * 60)
        
        for epoch in range(self.config['epochs']):
            epoch_start = time.time()
            
            print(f"\n📅 Epoch {epoch + 1}/{self.config['epochs']}")
            print("-" * 40)
            
            # Train
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics['loss'])
            
            # Validate
            val_metrics = self.validate()
            self.val_losses.append(val_metrics['loss'])
            
            # Update scheduler
            self.scheduler.step(val_metrics['loss'])
            
            # Print metrics
            epoch_time = time.time() - epoch_start
            print(f"\n   📊 Train Loss: {train_metrics['loss']:.4f} "
                  f"(cls={train_metrics['cls_loss']:.4f}, reg={train_metrics['reg_loss']:.4f})")
            print(f"   📊 Val Loss: {val_metrics['loss']:.4f} "
                  f"(cls={val_metrics['cls_loss']:.4f}, reg={val_metrics['reg_loss']:.4f})")
            print(f"   ⏱️ Time: {epoch_time:.1f}s")
            
            # Check for improvement
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
                self.epochs_without_improvement = 0
                print(f"   🎉 New best validation loss!")
            else:
                self.epochs_without_improvement += 1
            
            # Save checkpoint
            checkpoint_path = Path(self.config['output_dir']) / f"checkpoint_epoch{epoch + 1}.pt"
            self.save_checkpoint(str(checkpoint_path), is_best=is_best)
            
            # Early stopping
            if self.epochs_without_improvement >= self.config['early_stop_patience']:
                print(f"\n⛔ Early stopping after {epoch + 1} epochs")
                break
        
        print("\n" + "=" * 60)
        print("✅ Training complete!")
        print(f"   Best validation loss: {self.best_val_loss:.4f}")
        print("=" * 60)
        
        # Save final model
        final_path = Path(self.config['output_dir']) / "final_model.pt"
        self.save_checkpoint(str(final_path))
        print(f"   💾 Final model saved to {final_path}")
        
        return self.best_val_loss


# ============================================================================
# MAIN
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='Train SSD-CNN-256 for X-ray object detection')
    
    parser.add_argument('--data-root', type=str, default=DEFAULT_CONFIG['data_root'],
                        help='Path to processed data directory')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_CONFIG['output_dir'],
                        help='Directory to save models')
    parser.add_argument('--epochs', type=int, default=DEFAULT_CONFIG['epochs'],
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_CONFIG['batch_size'],
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=DEFAULT_CONFIG['learning_rate'],
                        help='Learning rate')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Update config with args
    config = DEFAULT_CONFIG.copy()
    config['data_root'] = args.data_root
    config['output_dir'] = args.output_dir
    config['epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.lr
    config['resume'] = args.resume
    
    # Train
    trainer = Trainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\n🔄 Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=config['device'], weights_only=False)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Pour le fine-tuning: NE PAS charger l'optimizer (on veut le nouveau LR)
        # Pour reprendre exactement: décommenter les lignes ci-dessous
        # if 'optimizer_state_dict' in checkpoint:
        #     trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'best_val_loss' in checkpoint:
            trainer.best_val_loss = checkpoint['best_val_loss']
            print(f"   📊 Previous best_val_loss: {trainer.best_val_loss:.4f}")
        
        if 'train_losses' in checkpoint:
            print(f"   📈 Previous training: {len(checkpoint['train_losses'])} epochs")
        
        print(f"   ✅ Model weights loaded (fine-tuning mode)")
        print(f"   🎯 New learning rate: {config['learning_rate']}")
    
    trainer.train()


if __name__ == "__main__":
    main()