import requests
import pandas as pd
import re
import os
import shutil

countries = ["Afghanistan", "Albania", "Algeria", "Andorra", "Angola", "Antigua and Barbuda",
             "Argentina", "Armenia", "Australia", "Austria", "Azerbaijan", "Bahamas", "Bahrain",
             "Bangladesh", "Barbados", "Belarus", "Belgium", "Belize", "Benin", "Bhutan", "Bolivia",
             "Bosnia and Herzegovina", "Botswana", "Brazil", "Brunei", "Bulgaria", "Burkina Faso",
             "Burundi", "Cabo Verde", "Cambodia", "Cameroon", "Canada", "Central African Republic",
             "Chad", "Chile", "China", "Colombia", "Comoros", "Congo", "Costa Rica", "Croatia",
             "Cuba", "Cyprus", "Czechia", "Côte d'Ivoire", "Denmark", "Djibouti",
             "Dominica", "Dominican Republic", "DR Congo", "Ecuador", "Egypt", "El Salvador",
             "Equatorial Guinea", "Eritrea", "Estonia", "Eswatini", "Ethiopia", "Fiji", "Finland",
             "France", "Gabon", "Gambia", "Georgia", "Germany", "Ghana", "Greece", "Grenada",
             "Guatemala", "Guinea", "Guinea-Bissau", "Guyana", "Haiti", "Holy See", "Honduras",
             "Hungary", "Iceland", "India", "Indonesia", "Iran", "Iraq", "Ireland", "Israel",
             "Italy", "Jamaica", "Japan", "Jordan", "Kazakhstan", "Kenya", "Kiribati", "Kuwait",
             "Kyrgyzstan", "Laos", "Latvia", "Lebanon", "Lesotho", "Liberia", "Libya",
             "Liechtenstein", "Lithuania", "Luxembourg", "Madagascar", "Malawi", "Malaysia",
             "Maldives", "Mali", "Malta", "Marshall Islands", "Mauritania", "Mauritius", "Mexico",
             "Micronesia", "Moldova", "Monaco", "Mongolia", "Montenegro", "Morocco", "Mozambique",
             "Myanmar", "Namibia", "Nauru", "Nepal", "Netherlands", "New Zealand", "Nicaragua",
             "Niger", "Nigeria", "North Korea", "North Macedonia", "Norway", "Oman", "Pakistan",
             "Palau", "Panama", "Papua New Guinea", "Paraguay", "Peru", "Philippines", "Poland",
             "Portugal", "Qatar", "Romania", "Russia", "Rwanda", "Saint Kitts & Nevis", "Saint Lucia",
             "Samoa", "San Marino", "Sao Tome & Principe", "Saudi Arabia", "Senegal", "Serbia",
             "Seychelles", "Sierra Leone", "Singapore", "Slovakia", "Slovenia", "Solomon Islands",
             "Somalia", "South Africa", "South Korea", "South Sudan", "Spain", "Sri Lanka",
             "St. Vincent & Grenadines", "State of Palestine", "Sudan", "Suriname", "Sweden",
             "Switzerland", "Syria", "Tajikistan", "Tanzania", "Thailand", "Timor-Leste", "Togo",
             "Tonga", "Trinidad and Tobago", "Tunisia", "Turkey", "Turkmenistan", "Tuvalu", "Uganda",
             "Ukraine", "United Arab Emirates", "United Kingdom", "United States", "Uruguay",
             "Uzbekistan", "Vanuatu", "Venezuela", "Vietnam", "Yemen", "Zambia", "Zimbabwe"]
try:
    os.mkdir("scraper_output/")
except:
    shutil.rmtree("scraper_output/")
    os.mkdir("scraper_output/")

for country in countries:
    try:
        os.mkdir("scraper_output/" + country)
    except:
        pass
    print(country)
    r = requests.get("https://www.google.com/search?tbm=isch&q=flag+of+" + country)
    index = r.text.find('src="http') + 5
    for i in range(len(r.text)):
        if r.text[index + i] == '"':
            end_index = index + i
            break
    img_url = r.text[index:end_index]
    print(img_url)
    img = requests.get(img_url)
    with open("scraper_output/" + country + "/" + country + ".bmp", "wb") as f:
        f.write(img.content)