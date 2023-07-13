# Python Programming for Data Science
# Python Exercises

# Görev 1: Verilen değerlerin veri yapılarını inceleyiniz.

x = 8
print(type(x))

y = 3.2
print(type(y))

z = 8j + 18
print(type(z))

a = "Hello World"
print(type(a))

b = True
print(type(b))

c = 23 < 22
print(type(c))

l = [1, 2, 3, 4]
print(type(l))

d = {"Name": "Jake",
     "Age": 27,
     "Adress": "Downtown"}
print(type(d))

t = ("Machine Learning", "Data Science")
print(type(t))

s = {"Python", "Machine Learning", "Data Science"}
print(type(s))


# Görev 2: Verilen string ifadenin tüm harflerini büyük harfe çeviriniz. Virgül ve nokta yerine space koyunuz,
# kelime kelime ayırınız

text = "The goal is to turn data into information, and information into insight."
print(text.upper().replace(",",".").split())


# Görev 3: Verilen listeye aşağıdaki adımları uygulayınız.

lst = ["D", "A", "T", "A", "S", "C", "I", "E", "N", "C", "E"]

# Adım 1: Verilen listenin eleman sayısına bakınız.
print(len(lst))

# Adım 2: Sıfırıncı ve onuncu indeksteki elemanları çağırınız.
print(lst[0]+ " " + lst[10])

# Adım 3: Verilen liste üzerinden ["D", "A", "T", "A"] listesi oluşturunuz.
new_list=lst[0:4]
print(new_list)

# Adım 4: Sekizinci indeksteki elemanı siliniz.
del(lst[8])
print(lst)

# Adım 5: Yeni bir eleman ekleyiniz.
lst.append('G')
print(lst)

# Adım 6: Sekizinci indekse "N" elemanını tekrar ekleyiniz.
lst.insert(8,'N')
print(lst)

# Görev 4: Verilen sözlük yapısına aşağıdaki adımları uygulayınız.

dict = {'Christian': ["America", 18],
        'Daisy': ["England", 12],
        'Antonio': ["Spain", 22],
        'Dante': ["Italy", 25]}

# Adım 1: Key değerlerine erişiniz.
print(dict.keys())

# Adım 2: Value'lara erişiniz.
print(dict.values())

# Adım 3: Daisy key'ine ait 12 değerini 13 olarak güncelleyiniz.
dict.update({'Daisy': ["England",13]})
print(dict)

# Adım 4: Key değeri Ahmet value değeri [Turkey,24] olan yeni bir değer ekleyiniz.
dict.update({'Ahmet': ["Turkey",24]})
print(dict)

# Adım 5: Antonio'yu dictionary'den siliniz.
del(dict['Antonio'])
print(dict)


# Görev 5: Argüman olarak bir liste alan, listenin içerisindeki tek ve çift sayıları ayrı listelere atayan ve bu listeleri
# return eden fonksiyon yazınız.

l = [2,13,18,93,22]
def func(list):

    even_list = []
    odd_list = []

    for i in list:
        if i % 2 == 0:
            even_list.append(i)
        else:
            odd_list.append(i)
    return even_list,odd_list

even,odd = func(l)
print(even)
print(odd)


# Görev 6: Aşağıda verilen listede mühendislik ve tıp fakülterinde dereceye giren öğrencilerin isimleri
# bulunmaktadır. Sırasıyla ilk üç öğrenci mühendislik fakültesinin başarı sırasını temsil ederken son üç öğrenci de
# tıp fakültesi öğrenci sırasına aittir. Enumarate kullanarak öğrenci derecelerini fakülte özelinde yazdırınız.

ogrenciler = ["Ali", "Veli", "Ayşe", "Talat", "Zeynep", "Ece"]

for i, ogrenci in enumerate(ogrenciler):
    if i<3:
        i += 1
        print("Mühendislik Fakültesi",i," . öğrenci:", ogrenci)
    else:
        i -=2
        print("Tıp Fakültesi",i, ". öğrenci:", ogrenci)


# Görev 7: Aşağıda 3 adet liste verilmiştir. Listelerde sırası ile bir dersin kodu, kredisi ve kontenjan bilgileri yer
# almaktadır. Zip kullanarak ders bilgilerini bastırınız.

ders_kodu = ["CMP1005", "PSY1001", "HUK1005", "SEN2204"]
kredi = [3,4,2,4]
kontenjan = [30,75,150,25]

for ders_kodu, kredi, kontenjan in zip(ders_kodu,kredi,kontenjan):
    print(f"Kredisi {kredi} olan {ders_kodu} kodlu dersin kontenjanı {kontenjan} dır.")


# Görev 8: Aşağıda 2 adet set verilmiştir. Sizden istenilen eğer 1. küme 2. kümeyi kapsiyor ise ortak elemanlarını
# eğer kapsamıyor ise 2. kümenin 1. kümeden farkını yazdıracak fonksiyonu tanımlamanız beklenmektedir.

kume1 = set(["data", "python"])
kume2 = set(["data", "function", "qcut", "lambda", "python", "miuul"])

if kume1.issuperset(kume2):
    print(kume1.intersection(kume2))
else:
    print(kume2.difference(kume1))
