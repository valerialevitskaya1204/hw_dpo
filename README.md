# hw_dpo

В качестве базовой модели брала Qwen 1.5B и датасет TinyStories

1. С LoRA forward немного дольше, потому что нужно умножить две матрицы A B и потом еще прибавить их к линейному слою.
Для обычного Qwen у меня в кагл не поместилась модель, поэтому backward я не считала (у меня постоянно была out of memory). По памяти модель с лорой занимает почти в два раза меньше места. Генерирует даже после трех эпох неплохо, но постоянно про какую-то Лили, других имен не выбирает, скорее всего особенность датасета, потому что он синтетический тоже  (нагенерен гптшкой)

https://www.kaggle.com/models/valery1891/qwen-lora тут квен после того как потюнила
https://www.kaggle.com/models/valery1891/pythia-lora тут pythia после того как натюнила

<img width="569" height="173" alt="image" src="https://github.com/user-attachments/assets/869975d8-33a3-4433-bcd2-c53d8104c62a" />
<img width="787" height="200" alt="image" src="https://github.com/user-attachments/assets/d6dd1e35-7ac5-4d5d-bcd2-3866f5ebc0f5" />

<img width="1165" height="721" alt="image" src="https://github.com/user-attachments/assets/02dfe5fd-3384-44d9-a2ac-d6f509744a94" />
<img width="484" height="695" alt="image" src="https://github.com/user-attachments/assets/dd59822d-a648-4e29-8d61-db5bf8f6ed2a" />

2. С pythia модель я обучила, но заинференсить не успела. Все скрипты обучения есть на гите
<img width="1102" height="681" alt="image" src="https://github.com/user-attachments/assets/06d904c8-c17b-4868-ad16-d9864b77063c" />


