address_templates = ['{region} обл., с.{village}, {street} {house_num}, поштовий код {index}',
                    '{index}, {city}, {street} {house_num}, кв.{flat_num} (район {district})',
                    '{index}, {region} обл.,{city}, просп.{avenue}, буд. {house_num}',
                    '{index}, {region} обл., {district} р-н, смт {village}, вул.{street}, буд. {house_num}, кв. {flat_num}',
                    '{region} обл., м.{city}, {avenue} просп., {house_num}, кв.{flat_num}',
                    '{region} обл., м.{city}, вул.{street}, буд. {house_num}, кв. {flat_num}',
                    '{index}, {region} обл., {city}, вул.{street}, буд. {house_num}',
                    '{index}, {region} обл., м.{city}, вул.{street}, {house_num}, кв.{flat_num}',
                    '{region} обл., {city}, {avenue} просп., {house_num}, кв.{flat_num}',
                    '{region} обл., м.{city}, просп.{avenue}, {house_num}, кв.{flat_num}',
                    '{index}, {region} обл.,{city}, {avenue} проспект,буд. {house_num},квартира {flat_num}',
                    '{index}, {region} обл., {city}, {avenue} проспект,буд. {house_num},квартира {flat_num}',
                    '{index}, {region} обл., м.{city}, вул.{street}, {house_num}, кв.{flat_num}',
                    '{index}, {region} обл.,р-н {district},с {village},просп.{avenue},буд. {house_num}',
                    'м.{city}, {district} район, вул.{street}, буд. {house_num}, кв. {flat_num}',
                    '{index}, {region} обл.,{city},проспект {avenue},буд. {house_num},квартира {flat_num}',
                    '{index}, {city},{avenue} проспект,буд. {house_num},квартира {flat_num}',
                    '{index}, {region} обл.,{city},просп.{avenue},буд. {house_num},квартира {flat_num}',
                    '{index}, {region} обл., {city},просп.{avenue},буд. {house_num},квартира {flat_num}',
                    '{index}, {region} обл., {city}, просп.{avenue},буд. {house_num},квартира {flat_num}',
                    '{region} обл., {district} р-н, {village}, вул.{street}, буд. {house_num}',
                    '{index}, м.{city}, вул.{street}, буд. {house_num}, кв. {flat_num}',
                    '{index}, {city},буд. {house_num},квартира {flat_num}, вул. {street}',
                    '{region} обл., {city}, просп.{avenue}, буд. {house_num}',
                    '{index}, {city},проспект {avenue},буд. {house_num},квартира {flat_num}',
                    '{region} обл., {district} р-н, с.{village}',
                    '{index}, м.{city}, {avenue} просп., {house_num}, кв.{flat_num}',
                    '{city}, {street}, {house_num}, {flat_num}',
                    '{region} обл., {city}, {avenue} просп., буд. {house_num}',
                    '{index}, {city}, {avenue} {house_num}, кв.{flat_num}',
                    '{region} обл., {district} р-н, {city}, вул.{street}, буд. {house_num}, кв. {flat_num}',
                    '{region} обл., {city}, просп.{avenue}, {house_num}, кв.{flat_num}',
                    '{index}, {region} обл.,р-н {district},{village},вул.{street},буд. {house_num}',
                    '{city}, вулиця {street}, буд. {house_num}/кв. {flat_num}',
                    '{index}, {region} обл.,р-н {district},с {village},{avenue} просп.,буд. {house_num}',
                    '{index}, {region} обл.,{city},просп.{avenue},буд. {house_num}, квартира {flat_num}',
                    '{region} обл., м.{city}, вул.{street}, {house_num}, кв.{flat_num}',
                    '{index}, м.{city}, вул.{street}, буд. {house_num}в, кв. {flat_num}',
                    '{index}, {region} обл., м.{city}, вул.{street}, кв. {flat_num}',
                    '{index}, {region} обл., м.{city}, пров.{street}, буд. {house_num}, кв. {flat_num}',
                    '{region} обл., {district} р-н, проспект {avenue}, буд. {house_num}, кв. {flat_num}',
                    '{city}, {district} р-н, вул.{street}, буд. {house_num}, кв.{flat_num}, {index}',
                    '{index}, {region} обл.,{city},смт {village},вул.{street},буд. {house_num},квартира {flat_num}',
                    '{region} обл., {district} р-н, с.{village}, вул.{street}, буд. {house_num}, кв. {flat_num}',
                    '{region} обл., {district} р-н, {street} {house_num}',
                    'с.{village}, {street} {house_num}, {region} обл., {index}',
                    '{index}, {region} обл., м.{city}, {avenue} просп, {house_num}',
                    '{index}, {region} обл., {district} р-н, с.{village}, вул.{street}, буд. {house_num}',
                    '{index}, {region} обл., м.{city}, смт {village}, вул.{street}, {house_num}',
                    '{index}, {region} обл., м.{city}, вул.{street}, буд. {house_num}б, кв. {flat_num}',
                    '{region} обл., {district} р-н, с.{village}, {street}, буд. {house_num}',
                    '{index}, {region} обл., м.{city}, {street} {house_num}, кв.{flat_num}',
                    '{region} обл., {district} р-н, с.{village}, вул.{street}',
                    '{index}, {city},провулок {street},буд. {house_num}, квартира {flat_num}',
                    '{index}, {city}, провулок {street},буд. {house_num}, квартира {flat_num}',
                    'м.{city}, {avenue}, просп. {house_num}, кв.{flat_num}',
                    '{index}, {region} обл.,{city},{avenue} просп.,буд. {house_num}',
                    '{index}, {region} обл.,{city},{avenue} просп.,буд. {house_num},квартира {flat_num}',
                    '{index}, {region} обл., {city},{avenue} просп.,буд. {house_num}',
                    '{index}, {region} обл., {city},{avenue} просп.,буд. {house_num},квартира {flat_num}',
                    '{index}, {region} обл., {city}, {avenue} просп.,буд. {house_num}, квартира {flat_num}',
                    '{index}, {region} обл.,{city}, {avenue} просп.,буд. {house_num}, квартира {flat_num}',
                    '{index}, {region} обл., {city}, {avenue} просп., буд. {house_num}, квартира {flat_num}',
                    'м.{city}, просп.{avenue}, {house_num}, кв.{flat_num}',
                    '{index}, {region} обл., {city},вул.{street}, буд. {house_num}',
                    '{index}, {city},проспект {avenue},буд. {house_num}',
                    '{region} обл., м.{city}, {street}, дім {house_num}',
                    '{index}, {city}, {street}, {house_num}',
                    '{index}, {region} обл., {city}, пров. {lane}, буд. {house_num}, кв.{flat_num}',
                    '{region} обл., м.{city}, {avenue} {house_num}, кв.{flat_num}',
                    '{index}, {region} обл., м.{city}, вул.{street}, {house_num}',
                    '{index}, {region} обл., {city}, {avenue} проспект,буд. {house_num}',
                    '{index}, {region} обл., {city}, проспект {avenue},буд. {house_num}',
                    '{index}, {region} обл.,{city},смт {village},вул.{street},буд. {house_num}',
                    '{index}, {region} обл., {city}, смт {village},вул.{street},буд. {house_num}',
                    '{city}, {street}, {house_num}',
                    '{region} обл., м.{city}, вул.{street}, {house_num}',
                    '{index}, {region} обл., {city}, вул.{street}, буд. {house_num}, кв. {flat_num}',
                    '{index}, {region} обл.,{district} р-н,{village},провулок {street},буд. {house_num}',
                    '{index}, {region} обл.,р-н {district},с {village},вул.{street},буд. {house_num}',
                    '{region} обл., {city}, вул.{street}, {house_num}, кв.{flat_num}',
                    '{index}, {region} обл., {district} р-н, с.{village}, вул.{street}, буд. {house_num}, кв. {flat_num}',
                    '{index}, {region} обл.,{city},пров.{street},буд. {house_num}, квартира {flat_num}',
                    '{index}, {region} обл., {city},пров.{street},буд. {house_num}, квартира {flat_num}',
                    '{index}, {region} обл., {city}, пров.{street}, буд. {house_num}, квартира {flat_num}',
                    '{index}, {region} обл.,вул.{street},буд. {house_num},квартира {flat_num}',
                    '{index}, {city},пров.{street},буд. {house_num},квартира {flat_num}',
                    '{index}, {city}, проспект {avenue}, будинок {house_num}, квартира {flat_num}',
                    '{index}, {region} обл., {district} р-н, {village}, {street} {house_num}, кв.{flat_num}',
                    '{index}, {region} обл.,{city},вул. {street},буд. {house_num},квартира {flat_num}',
                    '{index}, {region} обл., м.{city}, смт {village}, вул.{street}, буд. {house_num}',
                    '{index}, {region} обл.,{city},пров.{street},буд. {house_num},квартира {flat_num}',
                    '{index}, {region} обл., {district} р-н, м.{city}, пров.{street}, буд. {house_num}',
                    '{index}, {city},вул.{street},буд. {house_num}',
                    '{index}, {city},провулок {street},буд. {house_num},квартира {flat_num}',
                    '{index}, {region} обл.,{city},{avenue} просп.,буд. {house_num}, квартира {flat_num}',
                    '{index}, {region} обл., м.{city}, просп.{avenue}, {house_num}',
                    '{index}, м.{city}, бульвар {avenue}, буд. {house_num}, кв. {flat_num}',
                    '{index}, {city}, {avenue} проспект,буд. {house_num},квартира {flat_num}',
                    '{index}, {region} обл.,{city},вул.{street},буд. {house_num},квартира {flat_num}',
                    '{region} обл., {city}, вул.{street}, буд. {house_num}, кв.{flat_num}',
                    '{index}, {region} обл., {city}, пров.{street}, буд. {house_num}',
                    '{index}, {city}, вул.{street}, буд. {house_num}, кв. {flat_num}',
                    '{index}, {region} обл., {city},провулок {street},буд. {house_num}',
                    '{index}, {city},{avenue} проспект,буд. {house_num}',
                    '{region} обл., м.{city}, вул.{street}, буд. {house_num}',
                    '{index}, {region} обл., м.{city}, пл. {avenue}, буд. {house_num}, офіс {flat_num}',
                    '{index}, м.{city}, просп.{avenue}, {house_num}, кв.{flat_num}',
                    '{index}, м.{city}, вул.{street}, буд. {house_num}',
                    '{region} обл., м.{city}, вул.{street}, буд. {house_num}',
                    '{region} обл. {city}, {street}, {house_num}',
                    '{index}, {city},вул.{street},буд. {house_num},квартира {flat_num}',
                    '{index}, {city},вул. {street},буд. {house_num}, квартира {flat_num}',
                    '{index}, {region} обл.,{district} р-н,{village},вулиця {street},буд. {house_num}',
                    'м.{city}, вул.{street}, буд. {house_num}, кв. {flat_num}',
                    '{region} обл., {district} р-н, с.{village}, вул.{street}, буд. {house_num}',
                    '{index}, {region} обл., м.{city}, вул.{street}, буд. {house_num}, кв. {flat_num}',
                    '{index}, {region} обл., {district} р-н, {city}, вул.{street}, буд. {house_num}',
                    '{region} обл., {district} р-н, с.{village}, вул.{street}, {house_num}',
                    '{region} обл., {city}, вул.{street}, буд. {house_num}, кв. {flat_num}',
                    'м.{city}, вул.{street}, буд. {house_num}, кв.{flat_num}, {index} область {region}'
                    'м.{city}, вул.{street}, буд. {house_num}, кв.{flat_num} область {region}']

noises = [
    "-{noise}",
    "/{noise}",
    "{noise}.{noise}",
    ",{noise}",
    "{noise}",
    " -{noise}",
    " /{noise}",
    " {noise}.{noise}",
    " ,{noise}",
    " {noise}",
    "-{noise} ",
    "/{noise} ",
    "{noise}.{noise} ",
    ",{noise} ",
    "{noise} ",
    " -{noise} ",
    " /{noise} ",
    " {noise}.{noise} ",
    " ,{noise} ",
    " {noise} ",
]

ukrainian_alphabet = 'абвгґдеєжзиіїйклмнопрстуфхцчшщьюя`'
additional_chars = ' .,-/'
numbers = '0123456789'

allowed_chars = ukrainian_alphabet + additional_chars + numbers

avenues = ['Глибочицький',
      'Комсомольський',
      'Космонавтів',
      'Правобережний',
      'Корольова',
      'Білогірський',
      'Культури',
      'Свободи',
      'Івана Павла II',
      'Старомостицький',
      'Жовтневий',
      'Чернігівський',
      'Новокиївський',
      'Піонерський',
      'Ломоносова',
      'Патона',
      'Євпаторійський',
      'Українських Дивізій',
      'Пушкіна',
      "Солом'янський",
      'Скоропадського',
      'Старокиївський',
      'Шулявський',
      'Рівненський',
      'Олександра Довженка',
      'Херсонський',
      'Печерський',
      'Котляревського',
      'Івано-Франківський',
      'Керченський',
      'Калініна',
      'Харківський',
      'Оболонський',
      'Бахчисарайський',
      'Льва Толстого',
      'Теремківський',
      'Короленка',
      'Миколаївський',
      'Запорізький',
      'Тимошенко',
      'Полтавський',
      'Луганський',
      'Нивки',
      'Молодіжний',
      'Незалежності',
      'Житомирський',
      'Вінницький',
      'Київський',
      'Дружби Народів',
      'Януковича',
      'Робітничий',
      'Мазепи',
      'Берковецький',
      'Мистецтв',
      'Горького',
      'Правди',
      'Кучми',
      'Мечникова',
      'Кіровоградський',
      'Перемоги',
      'Кропивницький',
      'Одеський',
      'Соборний',
      'Литвина',
      'Кримський',
      'Судакський',
      'Чернівецький',
      'Хмельницький',
      'Центральний',
      'Чорновола',
      'Севастопольський',
      'Північний',
      'Ялтинський',
      'Ющенка',
      'Алчевський',
      'Донецький',
      'Індустріальний',
      'Закарпатський',
      'Юності',
      'Черкаський',
      'Шевченка',
      'Тернопільський',
      'Павлова',
      'Голосіївський',
      'Вячеслава Чорновола',
      'Сумський',
      'Феодосійський',
      'Героїв Небесної Сотні',
      'Технічний',
      'Дарницький',
      'Миру',
      'Романа Шухевича',
      'Південний',
      'Ярослава Мудрого',
      'Любомира Гузара',
      'Тараса Шевченка',
      'Кравчука',
      'Мостицький',
      'Західний',
      'Сімферопольський',
      'Львівський',
      'Зеленського',
      'Заводська',
      'Східний',
      'Володимира Великого',
      'Вічевий Майдан',
      'Івана Франка',
      'Богдана Хмельницького',
      'Подільський',
      '50-річчя Перемоги',
      'Кіпріана',
      'Сирецький',
      'Порошенка',
      'Волинський',
      'Лівобережний',
      'Липки',
      'Грушевського',
      'Петлюри',
      'Гагаріна',
      'Алуштинський',
      'Лесі Українки',
      'Василя Стуса',
      'Дніпровський',
      'Михайла Грушевського',
      'Видубицький',
      'Сталінський',
      'Бандери',
      'Сахарова',
      'Деснянський',
      'Симоненка']

avenues = [i.lower() for i in avenues]

cities = ['Житомир',
          'Жашків',
          'Суми',
          'Нова Каховка',
          'Пирятин',
          'Володимир-Волинський',
          'Вознесенськ',
          'Мелітополь',
          'Українка',
          'Жданівка',
          'Чугуїв',
          'Лисичанськ',
          'Вишгород',
          'Марганець',
          'Яремче',
          'Печерськ',
          'Іллічівськ',
          'Лозова',
          'Сміла',
          'Амвросіївка',
          'Ірпінь',
          'Фастів',
          'Тальне',
          'Яготин',
          'Вінниця',
          'Сарни',
          'Коростень',
          'Хмільник',
          'Дрогобич',
          'Золочів',
          'Горлівка',
          'Буча',
          'Чорноморськ',
          'Золотоноша',
          'Дніпрорудне',
          'Нікополь',
          'Полтава',
          'Ватутіне',
          'Ніжин',
          'Вишневе',
          'Євпаторія',
          'Дубно',
          'Острог',
          'Авдіївка',
          'Одеса',
          'Львів',
          'Рівне',
          'Вугледар',
          'Токмак',
          'Конотоп',
          'Миргород',
          "Слов'янськ",
          'Гайсин',
          'Білгород-Дністровський',
          'Київ',
          'Луганськ',
          'Здолбунів',
          'Павлоград',
          'Костопіль',
          'Нова Одеса',
          'Маріуполь',
          'Жовті Води',
          'Покровськ',
          'Славутич',
          'Запоріжжя',
          'Нетішин',
          'Новоград-Волинський',
          'Прилуки',
          'Чистякове',
          'Шепетівка',
          "Кам'янець-Подільський",
          'Жмеринка',
          'Енергодар',
          'Самбір',
          'Рені',
          'Краматорськ',
          'Ізюм',
          'Луцьк',
          'Хуст',
          'Бережани',
          'Коломия',
          'Мирноград',
          'Алчевськ',
          'Царичанка',
          'Глухів',
          'Шостка',
          'Кропивницький',
          'Бердянськ',
          "Кам'янське",
          'Біла Церква',
          'Сєвєродонецьк',
          'Скадовськ',
          'Шахтарськ',
          'Цюрупинськ',
          'Долина',
          'Хмельницький',
          'Дніпро',
          'Черкаси',
          'Бориспіль',
          'Ізмаїл',
          'Северодонецьк',
          'Кривий Ріг',
          'Харків',
          'Торецьк',
          'Обухів',
          'Вараш',
          'Костянтинівка',
          'Трускавець',
          'Лубни',
          'Горішні Плавні',
          'Калинівка',
          'Рожище',
          'Чернівці',
          'Керч',
          'Кадіївка',
          'Дебальцеве',
          'Сімферополь',
          'Овруч',
          'Харцизьк',
          'Миколаїв',
          'Новомосковськ',
          'Бахмут',
          'Білозерське',
          'Ковель',
          'Полонне',
          'Бердичів',
          'Феодосія',
          'Єнакієве',
          'Бровари',
          'Рубіжне',
          'Ялта',
          'Ужгород',
          'Олевськ',
          'Кременчук',
          'Тлумач',
          'Нововолинськ',
          'Івано-Франківськ',
          'Херсон',
          'Калуш',
          'Чернігів',
          'Монастириська',
          'Лебедин',
          'Тернопіль',
          'Мукачево',
          'Умань',
          'Первомайськ']

cities = [i.lower() for i in cities]

regions = [
    "Вінницька",
    "Волинська",
    "Дніпропетровська",
    "Донецька",
    "Житомирська",
    "Закарпатська",
    "Запорізька",
    "Івано-Франківська",
    "Київська",
    "Кіровоградська",
    "Луганська",
    "Львівська",
    "Миколаївська",
    "Одеська",
    "Полтавська",
    "Рівненська",
    "Сумська",
    "Тернопільська",
    "Харківська",
    "Херсонська",
    "Хмельницька",
    "Черкаська",
    "Чернівецька",
    "Чернігівська"
]

regions = [i.lower() for i in regions]

lanes = [
    "Академічний",
    "Береговий",
    "Веселковий",
    "Городоцький",
    "Дубовий",
    "Європейський",
    "Жасминовий",
    "Затишний",
    "Ігнатівський",
    "Казковий",
    "Лісовий",
    "Мальовничий",
    "Набережний",
    "Озерний",
    "Парковий",
    "Романтичний",
    "Садовий",
    "Тихий",
    "Уютний",
    "Фортунний",
    "Хвойний",
    "Центральний",
    "Чарівний",
    "Шовковий",
    "Щасливий",
    "Яблуневий",
    "Вишневий",
    "Гранітний",
    "Джерельний",
    "Єднання",
    "Журавлиний",
    "Зорепадний",
    "Інженерний",
    "Кленовий",
    "Лавандовий",
    "Медовий",
    "Новий",
    "Олімпійський",
    "Піонерський",
    "Річковий",
    "Сонячний",
    "Травневий",
    "Учительський",
    "Фіалковий",
    "Хмільний",
    "Цукровий",
    "Черешневий",
    "Шипшиновий",
    "Щедрий",
    "Ясеневий",
    "Виноградний",
    "Гірський",
    "Долинний",
    "Єлисейський",
    "Жовтневий",
    "Замковий",
    "Ірисовий",
    "Квітковий",
    "Луговий",
    "Місячний",
    "Нарцисовий",
    "Оптимістичний",
    "Перлинний",
    "Райдужний",
    "Скверний",
    "Тополиний",
    "Урожайний",
    "Фруктовий",
    "Хризантемовий",
    "Цирковий",
    "Чудовий",
    "Шахтарський",
    "Щитовий",
    "Ягідний",
    "Білий",
    "Вербовий",
    "Галявинний",
    "Дитячий",
    "Єгерський",
    "Жниварський",
    "Зелений",
    "Івовий",
    "Каштановий",
    "Лелековий",
    "Мирний",
    "Настурцієвий",
    "Осінній",
    "Плодовий",
    "Родинний",
    "Світанковий",
    "Теплий",
    "Успішний",
    "Фазанний",
    "Хоральний",
    "Царський",
    "Чародійний",
    "Шкільний",
    "Щогловий",
    "Ярмарковий"
]

lanes = [i.lower() for i in lanes]

districts = [
    "Шевченківський",
    "Печерський",
    "Солом'янський",
    "Оболонський",
    "Подільський",
    "Дарницький",
    "Дніпровський",
    "Деснянський",
    "Святошинський",
    "Голосіївський",
    "Сихівський",
    "Франківський",
    "Залізничний",
    "Личаківський",
    "Галицький",
    "Московський",
    "Київський",
    "Самбірський",
    "Бродівський",
    "Дрогобицький",
    "Червоноградський",
    "Стрийський",
    "Жовківський",
    "Яворівський",
    "Трускавецький",
    "Миколаївський",
    "Інгулецький",
    "Саксаганський",
    "Центрально-Міський",
    "Новобузький",
    "Первомайський",
    "Корабельний",
    "Херсонський",
    "Дніпровський",
    "Криворізький",
    "Суворовський",
    "Центральний",
    "Приморський",
    "Одеський",
    "Київський",
    "Малиновський",
    "Хмельницький",
    "Подільський",
    "Житомирський",
    "Богунський",
    "Корольовський",
    "Вінницький",
    "Замостянський",
    "Староміський",
    "Ленінський",
    "Кіровський",
    "Октябрський",
    "Жовтневий",
    "Радянський",
    "Автозаводський",
    "Комунарський",
    "Чернігівський",
    "Новозаводський",
    "Соснівський",
    "П'ятихатський",
    "Слобідський",
    "Новобаварський",
    "Холодногірський",
    "Шевченківський",
    "Комінтернівський",
    "Лівобережний",
    "Центральний",
    "Бабушкінський",
    "Амур-Нижньодніпровський",
    "Індустріальний",
    "Немишлянський",
    "Основ'янський",
    "Харківський",
    "Новобаварський",
    "Київський",
    "Салтівський",
    "Вовчанський",
    "Зміївський",
    "Куп'янський",
    "Лозівський",
    "Первомайський",
    "Харківський",
    "Чугуївський",
    "Балаклійський",
    "Барвінківський",
    "Богодухівський",
    "Валківський",
    "Великобурлуцький",
    "Вовчанський",
    "Дергачівський",
    "Зачепилівський",
    "Зміївський",
    "Золочівський",
    "Ізюмський",
    "Кегичівський",
    "Коломацький"
]
districts = [i.lower() for i in districts]

streets = [
    "Шевченка",
    "Грушевського",
    "Хрещатик",
    "Бандери",
    "Франка",
    "Пушкіна",
    "Соборна",
    "Лесі Українки",
    "Гагаріна",
    "Зелена",
    "Саксаганського",
    "Володимирська",
    "Рильського",
    "Льва Толстого",
    "Короленка",
    "Січових Стрільців",
    "Проспект Перемоги",
    "Дружби Народів",
    "Васильківська",
    "Мазепи",
    "Патона",
    "Київська",
    "Чорновола",
    "Липинського",
    "Стуса",
    "Котляревського",
    "Полтавська",
    "Симоненка",
    "Тичини",
    "Жукова",
    "Богуна",
    "Куліша",
    "Довженка",
    "Мельникова",
    "Кавказька",
    "Кирилівська",
    "Мечникова",
    "Артема",
    "Богдана Хмельницького",
    "Гончара",
    "Декабристів",
    "Європейська",
    "Житомирська",
    "Заболотного",
    "Івасюка",
    "Клименка",
    "Лепкого",
    "Мічуріна",
    "Некрасова",
    "Островського",
    "Петрівська",
    "Руставелі",
    "Садова",
    "Тарасівська",
    "Українка",
    "Філатова",
    "Хоткевича",
    "Ціолковського",
    "Чайковського",
    "Шолом-Алейхема",
    "Щорса",
    "Ярослава Мудрого",
    "Академіка Павлова",
    "Березняківська",
    "Вишенька",
    "Героїв Сталінграда",
    "Данила Галицького",
    "Єрмолова",
    "Жасминова",
    "Затишна",
    "Інженерна",
    "Казкова",
    "Лікарняна",
    "Мальовнича",
    "Нова",
    "Озерна",
    "Північна",
    "Революції",
    "Серпа і Молота",
    "Теплична",
    "Ушакова",
    "Фортечна",
    "Хімічна",
    "Центральна",
    "Червона",
    "Шкільна",
    "Щедра",
    "Яблунева"
]

streets = [i.lower() for i in streets]

villages = [
    "Петрівка",
    "Іванівка",
    "Степанівка",
    "Олександрівка",
    "Нова Слобода",
    "Зелене",
    "Веселе",
    "Сонячне",
    "Мирне",
    "Розсошенці",
    "Березове",
    "Дубове",
    "Лісове",
    "Польове",
    "Гранітне",
    "Калинівка",
    "Ясенівка",
    "Трояндове",
    "Червоне",
    "Синькове",
    "Вишневе",
    "Садове",
    "Білозерка",
    "Водяне",
    "Зоряне",
    "Кленове",
    "Лозове",
    "Надія",
    "Оріхове",
    "Піщане",
    "Річкове",
    "Соснове",
    "Тополине",
    "Урожайне",
    "Фруктове",
    "Хлібне",
    "Цукрове",
    "Чисте",
    "Шовкове",
    "Щасливе",
    "Яблуневе",
    "Берізка",
    "Ветеранське",
    "Городище",
    "Дружба",
    "Єдність",
    "Жовтень",
    "Заміське",
    "Іскра",
    "Квітуче",
    "Лугове",
    "Мальовниче",
    "Новобудова",
    "Оптимістичне",
    "Переможне",
    "Райдуга",
    "Світанок",
    "Тихе",
    "Успіх",
    "Фортуна",
    "Хороше",
    "Цвітуче",
    "Чудове",
    "Шепетівка",
    "Щедрий",
    "Ярмарок",
    "Білий Камінь",
    "Верховина",
    "Гірське",
    "Долина",
    "Животів",
    "Забуччя",
    "Іванків",
    "Красносілка",
    "Лелеківка",
    "Медове",
    "Новоселиця",
    "Олешня",
    "Пасічна",
    "Романів",
    "Солов'їна",
    "Трембачівка",
    "Ужинець",
    "Файне",
    "Хутори",
    "Царичанка",
    "Чарівне",
    "Шишківці",
    "Щурине",
    "Ясна Поляна"
]
villages = [i.lower() for i in villages]
