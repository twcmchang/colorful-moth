import os 
import tensorflow as tf
import time

def save_ckpt(step_patience_counter, model_dir, md, sess):
    if step_patience_counter == 0 :
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        checkpoint_path = os.path.join(model_dir, 'model_ckpt')
        md.saver_tf.save(sess, checkpoint_path) 
        print('------------[Model saved!]')
        return checkpoint_path

def earlystop(new_loss, current_best_val_loss, lr_patience_counter, step_patience_counter, min_delta = 0):
    if current_best_val_loss is None or current_best_val_loss - new_loss > min_delta:
        current_best_val_loss = new_loss
        lr_patience_counter = 0
        step_patience_counter = 0        
    else:
        step_patience_counter += 1
        lr_patience_counter += 1
    return current_best_val_loss, lr_patience_counter, step_patience_counter

def reduce_lr(l_rate, lr_patience_counter, lr_patience, lr_reduce_factor, min_lr = 0):
    if lr_patience_counter == lr_patience:
        l_rate *= lr_reduce_factor
        lr_patience_counter = 0
        print("------------[REDUCE learning rate] to %s" % (l_rate))
    if l_rate <= min_lr:
        l_rate = min_lr
    return l_rate, lr_patience_counter


class MyCallbacks():
    def __init__(self, args, model_dir, md, sess):
        self.args = args
        self.epoch_counter = 0
        self.epoch = args.epoch
        if args.test_mode == True:
            self.epoch = 10
        self.l_rate = args.lr
        self.current_best_val_loss = None
        self.min_delta = 0
        self.step_patience_counter = 0
        self.lr_patience_counter   = 0
        self.min_lr = 0
        self.earlystop_status = False
        self.model_dir = model_dir
        self.md = md
        self.sess = sess
        self.stime = time.time()
        
        
    def save_ckpt(self):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        checkpoint_path = os.path.join(self.model_dir, 'model_ckpt')
        self.md.saver_tf.save(self.sess, checkpoint_path) 
        print('------------[Model saved!]: %s\t' %(checkpoint_path))
        
    def earlystop(self):
        if self.step_patience_counter >= self.args.earlystop_patience:
            self.earlystop_status = True
            print('------------[Early Stopped: %s]\t' %(str(self.earlystop_status)))
        
    def reduce_lr(self):
        self.l_rate *= self.args.lr_reduce_factor
        self.lr_patience_counter = 0
        if self.l_rate <= self.min_lr:
            self.l_rate = self.min_lr
        print("------------[REDUCE Learning Rate]: to %s \t" % (self.l_rate))
        
    def print_out(self):
        print("\rEpoch[%s/%s] Pat[%s/%s] LR-Pat[%s/%s]---|train/valid-loss[%.4f/%.4f]|%s secs|l-rate:%s|"
      % (self.epoch_counter, self.epoch, 
         self.step_patience_counter, self.args.earlystop_patience,
         self.lr_patience_counter, self.args.lr_patience,
         self.train_loss, self.valid_loss, round(time.time()-self.stime,2), self.l_rate))
        self.stime = time.time()

    def update_record(self, train_loss, valid_loss, l_rate,
                      save_ckpt = False, earlystop = False,
                      reduce_lr = False, print_out = False):
        self.epoch_counter += 1
        self.l_rate = l_rate
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        if print_out ==True:
            self.print_out()
        if self.current_best_val_loss is None or self.current_best_val_loss - valid_loss > self.min_delta:
            self.current_best_val_loss = valid_loss
            if save_ckpt == True:
                self.save_ckpt()
            self.step_patience_counter = 0
            self.lr_patience_counter = 0
           
        else:
            self.step_patience_counter+=1
            if earlystop == True:
                self.earlystop()
            if reduce_lr == True:
                self.lr_patience_counter +=1
                if self.lr_patience_counter >= self.args.lr_patience:
                    self.reduce_lr()
                    
       
        return self.epoch_counter
            
    def get_current_lr(self):
        return self.l_rate
    
    def get_current_earlystop_status(self):
        return self.earlystop_status    
    def get_epoch(self):
        return self.epoch   
    
                
             
           
        
