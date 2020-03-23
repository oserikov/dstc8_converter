# dstc8-dstc2 converter

**usage example | иллюстрация запуска**: https://github.com/oserikov/dstc8_dstc2_converter/blob/master/dstc8_to_dstc2.ipynb
**converted data exapmles | примеры конвертированных данных**: https://drive.google.com/folderview?id=12pI20Wwv2RgdNKkEmR0_TNvMwt_UjjzJ

см. readme на русском ниже.

---
This repo contains utils used to convert the dstc8 dataset to the dstc2 format required by deeppavlov go-bot.
The single-domain dataset is outputted. The domain is configurable: you can choose any of domains used in dstc8 (see `DOMAIN_OF_INTEREST` in `constants.py`). 

**NB!** requires the dstc8 repo ([link](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue)) to be cloned 
(e.g. as it is done in the **usage example** provided above)

All the **configs** are set in `constants.py` .  
The most important ones are `DSTC8_REPO_PATH` and `DOMAIN_OF_INTEREST`.
The conversion results are saved to `BOT_DATA_DIR` and `DATA_DIR`.

4 converters are provided:
1. `naive_conversion.py` Converts the data without modifications of any kind. Mock api calls are added where it seemed to be relevant.(i.e. before *INFORM* and *NOTIFY* actions)  
   Is remarkably slow because of the *high level of linguistic variance* (lots of different phrases in the dataset do mean the same) and *high per-slot variance* (each slot has lots of candidates to be filled with e.g. lots of possible restaurant names) 
   The model training stops at **accuracy ~ 0.22** for the `Restaurants_1` domain.
2. `exp1_conversion.py` Converts the data. All the slots values are filled by mock values like `"slot_x_value"`. **Learns faster, no significant accuracy improvements**.  
3. `exp2_conversion.py` Converts the data, slots are filled with mock values as above, all the actions (e.g.  "Order_acceptance+Provide_phone_number") do now have the single way to be pronounced. Leans faster, achieves the **accuracy ~ 0.46** for `Restaurants_1` domain.
4. `exp3_conversion.py` Converts the data, slots are filled with mock values as above, action replics variance is reduced as above, aggregates the actions of the same type like this: "Provide_phone_number+Provide_address+Request_time" --> "Provide_needed+Request_needed". **accuracy became a bit worse** on the `Restaurants_1` domain.


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
