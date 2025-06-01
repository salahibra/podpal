import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pathlib import Path
import json

def summarize_chapters_and_global(chapters, model_path, output_path="summaries.json"):
    # Charger le modèle fine-tuné
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    summaries = []

    # Résumé par chapitre
    for chapter in chapters:
        inputs = tokenizer(
            chapter,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="longest"
        ).to(model.device)

        summary_ids = model.generate(
            **inputs,
            max_new_tokens=150,
            num_beams=4,
            no_repeat_ngram_size=3,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True
        )

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary.strip())

    # Résumé global à partir de tous les chapitres concaténés
    full_text = " ".join(chapters)
    inputs = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
        padding="longest"
    ).to(model.device)

    summary_ids = model.generate(
        **inputs,
        max_new_tokens=200,
        num_beams=4,
        no_repeat_ngram_size=3,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True
    )

    global_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Sauvegarder dans un fichier JSON
    output = {
        "chapter_summaries": summaries,
        "global_summary": global_summary.strip()
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"✅ Résumés sauvegardés dans : {output_path}")
    return output