import os
import cv2
import numpy as np

# Colore blu zaffiro (BGR)
SAPPHIRE = (255, 56, 0)

# Funzione per caricare l'immagine
def load_image(path):
    return cv2.imread(path)

# Funzione per salvare l'immagine
def save_image(path, image):
    cv2.imwrite(path, image)

# Funzione per ordinare i punti selezionati
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

# Funzione per la trasformazione prospettica
def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

# Funzione per selezionare i punti col mouse, con linee blu zaffiro, punti piccoli, drag e cancellazione

def select_points(image):
    points = []
    dragging_idx = None
    radius = 5
    thickness = 2
    win_name = "Seleziona 4 vertici"
    clone = image.copy()
    display = clone.copy()
    h, w = image.shape[:2]
    # Inizializza i punti ai 4 estremi
    points = [[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]]

    def redraw():
        nonlocal display
        display = clone.copy()
        # Disegna linee/frecce
        if len(points) > 1:
            for i in range(len(points)-1):
                color = (0, 165, 255) if i == 0 else SAPPHIRE  # Primo arco arancione, altri blu zaffiro
                cv2.arrowedLine(display, tuple(map(int, points[i])), tuple(map(int, points[i+1])), color, thickness, tipLength=0.08)
            if len(points) == 4:
                cv2.arrowedLine(display, tuple(map(int, points[3])), tuple(map(int, points[0])), SAPPHIRE, thickness, tipLength=0.08)
        # Disegna punti
        for idx, p in enumerate(points):
            cv2.circle(display, tuple(map(int, p)), radius, SAPPHIRE, -1)
            if idx == 0:
                cv2.circle(display, tuple(map(int, p)), radius+2, (0, 165, 255), 2)  # Primo punto evidenziato

    def get_nearest_idx(x, y):
        for i, p in enumerate(points):
            if np.linalg.norm(np.array([x, y]) - np.array(p)) < radius*2:
                return i
        return None

    def click_event(event, x, y, flags, param):
        nonlocal dragging_idx
        if event == cv2.EVENT_LBUTTONDOWN:
            idx = get_nearest_idx(x, y)
            if idx is not None:
                dragging_idx = idx
        elif event == cv2.EVENT_MOUSEMOVE and dragging_idx is not None:
            points[dragging_idx] = [x, y]
            redraw()
        elif event == cv2.EVENT_LBUTTONUP and dragging_idx is not None:
            points[dragging_idx] = [x, y]
            dragging_idx = None
            redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            idx = get_nearest_idx(x, y)
            if idx is not None:
                # Riporta il punto all'estremo originale
                if idx == 0:
                    points[idx] = [0, 0]
                elif idx == 1:
                    points[idx] = [w-1, 0]
                elif idx == 2:
                    points[idx] = [w-1, h-1]
                elif idx == 3:
                    points[idx] = [0, h-1]
                redraw()

    redraw()
    cv2.imshow(win_name, display)
    cv2.setMouseCallback(win_name, click_event)
    print("\nLegenda:")
    print("- Trascina un vertice: tasto sinistro del mouse")
    print("- Riporta un vertice all'estremo originale: tasto destro del mouse sul punto")
    print("- Conferma: INVIO")
    print("- Esci senza modificare: ESC")
    while True:
        cv2.imshow(win_name, display)
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Invio per confermare
            break
        elif key == 27:  # ESC per annullare
            break
    cv2.destroyWindow(win_name)
    return np.array(points, dtype="float32")

# Main

def process_image(image, pts=None):
    if pts is None:
        pts = select_points(image)
    return four_point_transform(image, pts)

def main():
    try:
        dir_path = input("Inserisci il percorso della directory con le immagini: ") or "./test_receipt"
        files = [f for f in os.listdir(dir_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        for fname in files:
            img_path = os.path.join(dir_path, fname)
            image = load_image(img_path)
            print(f"\nImmagine: {fname}")
            coords = input("Inserisci le 4 coordinate (x1,y1,x2,y2,x3,y3,x4,y4) o premi invio per selezionare col mouse: ")
            if coords.strip():
                pts = np.array(list(map(float, coords.split(","))), dtype="float32").reshape(4, 2)
            else:
                pts = None
            warped = process_image(image, pts)
            cv2.imshow("Risultato", warped)
            save_path = os.path.join(dir_path, f"cropped_{fname}")
            # save_image(save_path, warped)
            # print(f"Immagine salvata: {save_path}")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    except KeyboardInterrupt:
        print("\nTerminazione richiesta dall'utente (Ctrl+C). Uscita...")
        cv2.destroyAllWindows()
        exit(0)

if __name__ == "__main__":
    main()

