"""
Fine-tune E5 embedding model on NBA Q&A pairs using contrastive learning.
"""

import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
from sentence_transformers import SentenceTransformer, losses, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NBAQADataset(Dataset):
    """Dataset for NBA Q&A pairs"""
    
    def __init__(self, pairs, tokenizer, max_length=128):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # E5 requires "query: " prefix for questions and "passage: " for contexts
        question = "query: " + pair['question']
        context = "passage: " + pair['context']
        
        question_encoded = self.tokenizer(
            question,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        context_encoded = self.tokenizer(
            context,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'question_input_ids': question_encoded['input_ids'].squeeze(),
            'question_attention_mask': question_encoded['attention_mask'].squeeze(),
            'context_input_ids': context_encoded['input_ids'].squeeze(),
            'context_attention_mask': context_encoded['attention_mask'].squeeze(),
        }


def mean_pooling(model_output, attention_mask):
    """Mean pooling for sentence embeddings"""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class ContrastiveLossTrainer(Trainer):
    """Custom trainer with contrastive loss"""
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # Get question embeddings
        question_outputs = model(
            input_ids=inputs['question_input_ids'],
            attention_mask=inputs['question_attention_mask']
        )
        question_embeddings = mean_pooling(question_outputs, inputs['question_attention_mask'])
        question_embeddings = torch.nn.functional.normalize(question_embeddings, p=2, dim=1)
        
        # Get context embeddings
        context_outputs = model(
            input_ids=inputs['context_input_ids'],
            attention_mask=inputs['context_attention_mask']
        )
        context_embeddings = mean_pooling(context_outputs, inputs['context_attention_mask'])
        context_embeddings = torch.nn.functional.normalize(context_embeddings, p=2, dim=1)
        
        # Contrastive loss (InfoNCE)
        # Positive pairs should have high similarity, negatives should have low similarity
        batch_size = question_embeddings.shape[0]
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(question_embeddings, context_embeddings.T)
        
        # Temperature scaling
        temperature = 0.05
        similarity_matrix = similarity_matrix / temperature
        
        # Labels: diagonal elements are positive pairs
        labels = torch.arange(batch_size).to(similarity_matrix.device)
        
        # Cross entropy loss
        loss = torch.nn.functional.cross_entropy(similarity_matrix, labels)
        
        return (loss, (question_outputs, context_outputs)) if return_outputs else loss


def finetune_e5_model(training_pairs, validation_pairs, output_dir='./finetuned_e5'):
    """Fine-tune E5 model using contrastive learning"""
    
    logger.info("Loading E5 base model and tokenizer...")
    model_name = 'intfloat/e5-base-v2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Prepare datasets
    logger.info("Preparing datasets...")
    train_dataset = NBAQADataset(training_pairs, tokenizer)
    val_dataset = NBAQADataset(validation_pairs, tokenizer)
    
    # Training configuration
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=50,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        seed=42,
    )
    
    # Log hyperparameters
    hyperparams = {
        "model": model_name,
        "num_epochs": training_args.num_train_epochs,
        "batch_size": training_args.per_device_train_batch_size,
        "learning_rate": training_args.learning_rate,
        "weight_decay": training_args.weight_decay,
        "warmup_steps": training_args.warmup_steps,
        "temperature": 0.05,
        "max_length": 128,
        "training_samples": len(training_pairs),
        "validation_samples": len(validation_pairs),
        "timestamp": datetime.now().isoformat()
    }
    
    with open(f'{output_dir}/hyperparameters.json', 'w') as f:
        json.dump(hyperparams, f, indent=2)
    
    logger.info(f"Training configuration: {hyperparams}")
    
    # Create trainer
    trainer = ContrastiveLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    # Train
    logger.info("Starting training...")
    train_result = trainer.train()
    
    # Save model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Save training metrics
    metrics = train_result.metrics
    with open(f'{output_dir}/training_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info("Training complete!")
    return trainer, model, tokenizer


def finetune_with_sentence_transformers(training_pairs, validation_pairs, output_dir='./finetuned_e5_st'):
    """
    Alternative fine-tuning approach using sentence-transformers library.
    This is often easier and more robust.
    """
    logger.info("Fine-tuning with sentence-transformers library...")
    
    # Load model
    model = SentenceTransformer('intfloat/e5-base-v2')
    
    # Prepare training examples
    train_examples = []
    for pair in training_pairs:
        # Add prefixes as required by E5
        question = "query: " + pair['question']
        context = "passage: " + pair['context']
        # InputExample expects (texts, label) where label=1 for similar pairs
        train_examples.append(InputExample(texts=[question, context], label=1.0))
    
    # Prepare validation examples
    val_examples = []
    val_sentences1 = []
    val_sentences2 = []
    val_scores = []
    
    for pair in validation_pairs:
        question = "query: " + pair['question']
        context = "passage: " + pair['context']
        val_sentences1.append(question)
        val_sentences2.append(context)
        val_scores.append(1.0)
    
    # Create evaluator
    evaluator = EmbeddingSimilarityEvaluator(
        val_sentences1,
        val_sentences2,
        val_scores,
        name='nba-qa-validation'
    )
    
    # Create dataloader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)
    
    # Define loss function
    train_loss = losses.CosineSimilarityLoss(model)
    
    # Training configuration
    num_epochs = 10
    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)
    
    hyperparams = {
        "model": "intfloat/e5-base-v2",
        "num_epochs": num_epochs,
        "batch_size": 8,
        "warmup_steps": warmup_steps,
        "training_samples": len(training_pairs),
        "validation_samples": len(validation_pairs),
        "loss_function": "CosineSimilarityLoss",
        "timestamp": datetime.now().isoformat()
    }
    
    with open(f'{output_dir}/hyperparameters.json', 'w') as f:
        json.dump(hyperparams, f, indent=2)
    
    logger.info(f"Training configuration: {hyperparams}")
    
    # Train
    logger.info("Starting training...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        evaluator=evaluator,
        evaluation_steps=50,
        output_path=output_dir,
        show_progress_bar=True,
        save_best_model=True
    )
    
    logger.info(f"Model saved to {output_dir}")
    return model


if __name__ == "__main__":
    # Load data
    with open('training_data.json', 'r') as f:
        training_pairs = json.load(f)
    
    with open('validation_data.json', 'r') as f:
        validation_pairs = json.load(f)
    
    logger.info(f"Loaded {len(training_pairs)} training pairs")
    logger.info(f"Loaded {len(validation_pairs)} validation pairs")
    
    # Method 1: Using sentence-transformers (Recommended - easier and more robust)
    logger.info("\n=== Method 1: Fine-tuning with sentence-transformers ===")
    finetuned_model = finetune_with_sentence_transformers(
        training_pairs,
        validation_pairs,
        output_dir='./finetuned_e5_st'
    )
    
    # Method 2: Custom implementation (Optional - more control but more complex)
    # logger.info("\n=== Method 2: Custom fine-tuning implementation ===")
    # trainer, model, tokenizer = finetune_e5_model(
    #     training_pairs,
    #     validation_pairs,
    #     output_dir='./finetuned_e5_custom'
    # )