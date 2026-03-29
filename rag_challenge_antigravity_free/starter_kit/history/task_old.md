# RAG Challenge — Task Tracker

## Фаза 1: Подготовка
- [x] 1.1 Выбор фреймворка и бюджета
- [x] 1.2 Настройка окружения (venv, зависимости)
- [x] 1.3 Скачать документы и вопросы
- [x] 1.4 Анализ корпуса и вопросов
- [x] 1.5 Первый наивный pipeline
- [x] 1.6 Baseline submission (ждём подтверждения)

## Фаза 2: PDF-обработка (ингест)
- [x] 2.1 OCR для сканов
- [x] 2.2 Извлечение метаданных
- [x] 2.3 Умный chunking
- [x] 2.4 Маппинг chunk → (doc_id, pages)

## Фаза 3: Поиск
- [/] 3.1 Hybrid search (BM25 + vector)
- [ ] 3.2 Увеличить top-k
- [ ] 3.3 Re-ranking
- [ ] 3.4 Query expansion

## Фаза 4: Генерация
- [ ] 4.1 Промпты по answer_type
- [ ] 4.2 null-обработка
- [ ] 4.3 Post-processing

## Фаза 5: Grounding-оптимизация
- [x] 5.1 Фильтрация страниц через LLM Attribution
- [x] 5.2 Парсинг ответа (списки страниц, null)
- [x] 5.3 Streaming для фикса TTFT
- [x] 5.4 Восстановление файла submission_phase5_recreated.json

## Фаза 6: Телеметрия и скорость
- [x] 6.1 TTFT fix (стриминг токенов)
- [ ] 6.2 Оптимизация TTFT

## Фаза 7: Оценка OCR и сравнение LLM
- [/] 7.1 Сравнение DeepSeek vs GPT-4o-mini via OpenRouter
- [/] 7.2 Аудит качества извлечения текста (OCR)
- [/] 7.3 Hybrid Search (BM25 + Vector)
## Фаза 10: PDF Ingestion & OCR Audit
- [x] 10.1 Сравнение PyMuPDF, pdfplumber (Docling пропущен из-за сети)
- [ ] 10.2 Замер CER/WER на выборке документов

## Фаза 11: Embeddings & Advanced Retrieval
- [x] 11.1 Реализация Semantic Chunking
- [x] 11.2 Тест BGE-large vs E5 vs bge-small (Выбран bge-small для скорости)
- [x] 11.3 Тюнинг Hybrid Search & Reranker

## Фаза 12: LLM Comparison
- [/] 12.1 Сравнение DeepSeek, GPT-4o-mini, Claude 3.5 (GPT-4o-mini лидирует)
- [ ] 12.2 Настройка LLM-Judge (Faithfulness/Groundedness)

## Фаза 13: Quality Control & Testing
- [x] 13.1 Скрипт валидации submission (null check, retrieval check)
- [/] 13.2 Валидация и корректировка форматов (dates/numbers/lists)

## Фаза 14: Final Comparison & Submission
- [x] 14.1 Генерация submission_comparison_final.csv
- [x] 14.2 Финальный выбор и сохранение submission_final.json
## Фаза 15: Deep Quality Cleanup & Accuracy
- [x] 15.1 Metadata-enriched Prompt (Manifest)
- [x] 15.2 Grounding Citation Filter
- [x] 15.3 Length Guard (<280) & Quote Fix
- [x] 15.4 Reflection/Verification Loop
## Фаза 20: Forensic RAG Analysis - Question #2
- [x] 20.1 Isolated Debugging for Q#2
- [x] 20.2 Case-ID Strict Filtering Implementation
- [x] 20.3 Mandatory ID Match Prompt logic
- [x] 20.4 Verification of 2026-02-02 (SUCCESS)
- [/] 20.5 Full 100-question run and regression audit
