import re

# Test text with existing wort schatz sections in the middle
test_text = """Heldmaschine ist eine deutsche Band!
wort schatz
heldmaschine = kahraman makinesi 😊

Sie machen Musik, die NDH heißt. wort schatz
heißt = denir

Ich habe im Internet nachgeschaut. wort schatz
internet = internet

*   Sie wurde 2008 gegründet. 👍
*   Sie kommt aus Koblenz. Magst du ihre Musik? 🦊🔧"""

print("=== ORIGINAL TEXT ===")
print(test_text)

# Use a more comprehensive approach to clean the text
# Split the text into lines
lines = test_text.split('\n')
cleaned_lines = []

# Process each line
i = 0
while i < len(lines):
    line = lines[i].strip()

    # Skip 'wort schatz' lines and their associated translations
    if line.lower() == 'wort schatz':
        # Skip this line and any following translation lines (format: word = translation)
        i += 1
        while i < len(lines) and '=' in lines[i]:
            i += 1
        continue

    # Skip lines that end with 'wort schatz' and the following line if it contains '='
    if line.lower().endswith('wort schatz'):
        i += 1
        if i < len(lines) and '=' in lines[i]:
            i += 1
        continue

    # Skip standalone translation lines (format: word = translation)
    if '=' in line and len(line.split('=')) == 2:
        word_part = line.split('=')[0].strip().lower()
        # Check if this looks like a translation line
        if len(word_part) > 0 and not line.startswith('*') and not line.startswith('-'):
            i += 1
            continue

    # Add the line to our cleaned lines
    cleaned_lines.append(line)
    i += 1

# Rejoin the cleaned lines
cleaned_text = '\n'.join(cleaned_lines)

print("\n=== CLEANED TEXT ===")
print(cleaned_text)

# Add translations at the end
translations = {
    "heldmaschine": "kahraman makinesi",
    "heißt": "denir",
    "internet": "internet",
    "gegründet": "kuruldu"
}

result = cleaned_text + "\n\nwort schatz"
for word, translation in translations.items():
    result += f"\n{word} = {translation}"
result += " 😊"

print("\n=== FINAL RESULT ===")
print(result)
