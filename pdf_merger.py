from pypdf import PdfWriter, PdfReader

pdfs = [
    "kazusy_baza/Kazus z egzaminu z 17 czerwca 2024 r. wraz z rozwiązaniem.pdf",
    "kazusy_baza/Kazus z egzaminu z 17-6-2025.pdf",
    "kazusy_baza/Kazus z egzaminu z 18 czerwca 2024 r. wraz z rozwiązaniem.pdf",
    "kazusy_baza/Kazus z egzaminu z prawa cywilnego z 16-06-2025.pdf",
    "kazusy_baza/Kazus z konkursu z 2025.pdf",
    "kazusy_baza/Kazus z konkursu z prawa cywilnego (9.05.2024).pdf",
    "kazusy_baza/kazus_3092024.pdf",
    "kazusy_baza/Kazus-z-2-09-2025.pdf"
]

writer = PdfWriter()

for pdf in pdfs:
    reader = PdfReader(pdf)
    for page in reader.pages:
        writer.add_page(page)

with open("kazusy_rozwiazania.pdf", "wb") as f:
    writer.write(f)

print("Połączono PDF-y.")