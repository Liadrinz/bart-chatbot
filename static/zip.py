from PIL import Image


Image.open("bart.png").resize((50, 70)).save("bart-mini.png")
Image.open("bart-fast.png").resize((50, 70)).save("bart-fast-mini.png")
Image.open("icebear.png").resize((50, 50)).save("icebear-mini.png")
