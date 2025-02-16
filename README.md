 Simple Model Distillation: Making GPT-2 Smarter

This project makes a smaller language model (GPT-2 124M) smarter by teaching it with a bigger model (GPT-Neo-2.7B). We use the HellaSwag score metric to see if it worked.

## What We Did

We used "knowledge distillation." Think of it like a teacher (the big model) helping a student (the small model) learn.

*   **Teacher:** GPT-Neo-2.7B (big and smart)
*   **Student:** GPT-2 124M (smaller, we want to improve it)
*  Both models sizes are documented.

## The Test: HellaSwag

HellaSwag is a test that checks if a model understands common sense. It gives the model a sentence and some choices for how to finish it. Only one choice makes sense.

*   **Our Goal:** Beat the original GPT-2's score on HellaSwag, which was **0.2955**. Higher score = better.

## How Distillation Works (Simplified)

1.  **Teacher's Answers:** The big model (GPT-Neo-2.7B) is used to generate "soft targets" on the **WikiText-2** dataset.  This means we get its predictions, *including how sure it is* about each word (not just the best one).
2.  **Student Learns:** The small model (GPT-2) is trained on **WikiText-2** in two ways:
    *   It tries to predict the correct next word in the text (like normal language model training).
    *   It also tries to *match how sure* the teacher was about each word. This is the "distillation" part.  It learns from the teacher's "soft targets."
3.  **Softening the Answers:** The answers (called "logits") from both models are divided by a temperature value.  The higher the temperature, the "softer" (less confident) the predictions become.  This helps the student learn.

