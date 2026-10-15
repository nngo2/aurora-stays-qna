# Property names module shared by chunking (index time), retrieval (query time), and the backend
import re

# A property matrix row after the tables are linearized, e.g. "Property: ASH-DEN Union, Denver CO |"
PROPERTY_ROW = re.compile(r"Property: (ASH-[A-Z]{3}) ([^,|]+?), ([^|]+?) ([A-Z]{2}) \|")

# A row that names the property by code and name only, e.g. "Property: ASH-DEN Union |" (Refunds guide table)
CODE_ONLY_ROW = re.compile(r"Property: (ASH-[A-Z]{3}) ([^,|]+?) \|")

def build_property_map(texts):
    # Property code -> (name, city, state), parsed from the matrix rows, so no property list is hard-coded
    properties = {}
    for text in texts:
        for code, name, city, state in PROPERTY_ROW.findall(text):
            properties[code] = (name.strip(), city.strip(), state)
    return properties

def add_city_to_rows(text, properties):
    # Index time: "Property: ASH-DEN Union |" -> "Property: ASH-DEN Union, Denver CO |" (the matrix row form)
    def with_city(match):
        code = match.group(1)
        if code not in properties:
            return match.group(0)
        name, city, state = properties[code]
        return f"Property: {code} {name}, {city} {state} |"
    return CODE_ONLY_ROW.sub(with_city, text)

def add_property_codes(query, properties):
    # Query time: append "code name" for every property the question names by city or property name but not
    # by code, e.g. "the Denver hotel" -> "... (ASH-DEN Union)". City matches ignore case; names must be
    # capitalized, because names such as "Loop" or "Union" are also ordinary words.
    found = [
        f"{code} {name}"
        for code, (name, city, _state) in properties.items()
        if code.lower() not in query.lower()
        and (re.search(rf"\b{re.escape(city)}\b", query, re.IGNORECASE) or re.search(rf"\b{re.escape(name)}\b", query))
    ]
    return f"{query} ({'; '.join(found)})" if found else query
