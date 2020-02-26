# dstc8-dstc2 converter

код с иллюстрацией запуска: https://colab.research.google.com/drive/15snXsuxOVwS6XdOjB4ULyc8HKlhlUF-u

примеры конвертированных данных: todo 9.30 AM

---

Конвертер датасета dstc8 в формат simple-dstc2, читаемый go_bot-ом deeppavlov.
Конвертер формирует single-domain датасет в формате simple-dstc2, домен можно выбрать любой из тех, что есть в dstc8 (но тестировалось на паре-тройке)

**NB!** требует (правильного конфига:) ) наличия датасета dstc8 где-то (тестился только при наличии репо dstc8 в корне этого репо, но влиять не должно), см. пример запуска.

Ссылка на датасет dstc8: [ссылка](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue)

**Конфиги** в `constants.py` .  
Самые интересные, кажется, `DSTC8_REPO_PATH` и `DOMAIN_OF_INTEREST`.
Сохраняет полученные в процессе работы данные в `BOT_DATA_DIR` и `DATA_DIR`

Релизованы 4 конвертера:
1. `naive_conversion.py` Конвертирует данные, не изменяя их смысл. Добавляет заглушку с вызовом api там, где это показалось релевантным (перед действиями INFORM и NOTIFY)  
   Работает медленно по причине большого количества кандидатов на каждый слот и большого количества альтернативных вариантов реплик системы.  
   Обучение (очень! медленное) останавливается на **accuracy ~ 0.22** для домена `Restaurants_1`
2. `exp1_conversion.py` Конвертирует данные, но значения всех слотов заменяет на заглушки в виде "значение_слота_x". **Обучается быстрее, accuracy особенно не меняется**.  
3. `exp2_conversion.py` Конвертирует данные, значения всех слотов по-прежнему заменяет на заглушки, для каждого типа действия (например "Подтвердить_заказ+Озвучить_телефон") оставляет в датасете только один вариант соответствующей реплики, заменяя другие варианты реплик этого действия на заранее выбранный. Обучение (достаточно быстрое) останавливается на **accuracy ~ 0.46** для домена `Restaurants_1`
4. `exp3_conversion.py` Конвертирует данные, значения всех слотов по-прежнему заменяет на заглушки, по-прежнему убирает вариативность реплик, считает одинаковыми все действия одного класса, заменяя, например "Озвучить_X+Озвучить_Y+Запросить_Z" на "Озвучить_требуемое+Запросить_требуемое". Идея такая себе -- **accuracy хуже на несколько сотых** для домена `Restaurants_1`
