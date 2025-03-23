from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
)
from sentence_transformers import losses
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator


def sentence_transformer_train(args, dataset):
    model = SentenceTransformer(args.pretrain_model_path)

    train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']

    # Inputs,Labels,Appropriate Loss Functions
    # "(sentence_A, sentence_B) pairs",class,SoftmaxLoss
    # "(anchor, positive) pairs",none,MultipleNegativesRankingLoss
    # "(anchor, positive/negative) pairs","1 if positive, 0 if negative","ContrastiveLoss / OnlineContrastiveLoss"
    # "(sentence_A, sentence_B) pairs",float similarity score,"CoSENTLoss / AnglELoss / CosineSimilarityLoss"
    # "(anchor, positive, negative) triplets",none,"MultipleNegativesRankingLoss / TripletLoss"
    dev_evaluator = None
    if args.target_type == "a_b_score":
        train_loss = losses.CosineSimilarityLoss(model=model)
        dev_evaluator = EmbeddingSimilarityEvaluator(
            sentences1=eval_dataset["sentence1"],
            sentences2=eval_dataset["sentence2"],
            scores=eval_dataset["score"],
            name=f"{args.run_name}_dev"
        )
        dev_evaluator(model)
    elif args.target_type == "a_b_class":
        train_loss = losses.SoftmaxLoss(model=model)
    elif args.target_type == "anchor_p":
        train_loss = losses.MultipleNegativesRankingLoss(model=model)
    elif args.target_type == "anchor_pn":
        train_loss = losses.ContrastiveLoss(model=model)
    elif args.target_type == "anchor_p_n":
        train_loss = losses.TripletLoss(model=model)
    else:
        train_loss = None

    train_args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir=args.model_save_path,
        # Optional training parameters:
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        warmup_ratio=args.warmup_ratio,
        fp16=args.fp16,  # Set to False if your GPU can't handle FP16
        bf16=args.bf16,  # Set to True if your GPU supports BF16
        # Optional tracking/debugging parameters:
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        run_name=args.run_name,  # Used in W&B if `wandb` is installed
    )

    if dev_evaluator:
        trainer = SentenceTransformerTrainer(
            model=model,
            args=train_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss=train_loss,
            evaluator=dev_evaluator
        )
    else:
        trainer = SentenceTransformerTrainer(
            model=model,
            args=train_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss=train_loss
        )
    trainer.train()

    # test_evaluator = EmbeddingSimilarityEvaluator(
    #     sentences1=test_dataset["sentence_A"],
    #     sentences2=test_dataset["sentence_B"],
    #     scores=test_dataset["similarity_score"],
    #     name=f"{args.run_name}_test"
    # )
    # test_evaluator(model)

    model.save_pretrained(args.model_save_path)
