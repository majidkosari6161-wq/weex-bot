# SuperBot v0.1.0

سیستم معاملاتی با Backtest + یادگیری ماشین + بهینه‌سازی Walk-Forward
برای WEEX Futures Demo.

ساخته شده روی weex_v167_8_14_DEMO_EXEC.py (منطق entry/exit دست‌نخورده).

## شامل چه چیزهایی است

- Backtest Engine — بک‌تست دقیقاً مثل live کار می‌کنه
- ML/AI — مدل LightGBM با ۲۷ ویژگی
- WFO — بهینه‌سازی پارامترها با Optuna
- Reporting — گزارش HTML + JSON + Markdown
- Trade Ledger — ذخیره‌ی کامل معاملات در Parquet
- CLI — دستور python -m superbot

## نصب

pip install -r requirements.txt

نیاز به Python 3.11+

## شروع سریع

python -m superbot status
python -m superbot backtest --start 2026-08-17T00:00:00Z --end 2026-09-16T00:00:00Z

## نتیجه ۹۰ روز بک‌تست

با fee=0:  سود +8.86%
با fee=0.05% (WEEX): ضرر -2.30%

نتیجه: استراتژی قبل از fee سودآوره، ولی fee صرافی سودش رو از بین می‌بره.

## راه‌حل‌های پیشنهادی

- استفاده از maker orders (0.02% به جای 0.06%)
- کاهش leverage از 5x به 3x
- تایم‌فریم بالاتر (1H به جای 15m)
- فیلتر entry با مدل ML

## ساختار پوشه‌ها

superbot/backtest/  → engine, sim, clock
superbot/ml/        → features, labeler, trainer, filter
superbot/wfo/       → optimizer
superbot/reporting/ → report, ledger
superbot/commands/  → CLI

## محدودیت‌ها

- Isotonic calibration غیرفعاله (روی داده کم خراب می‌شه)
- مدل ML فقط با ۹۰ روز داده train شده
- دمو WEEX از openOrders پشتیبانی نمی‌کنه
- استراتژی با fee واقعی WEEX سودآور نیست

## لایسنس

اختصاصی. فقط استفاده داخلی.