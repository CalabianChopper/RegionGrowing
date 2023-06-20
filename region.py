import cv2
import numpy as np

#per il pacchetto - "pip install opencv-python"

def region_growing(image, seed):
    height, width = image.shape[:2]
    
    visited = np.zeros((height, width), dtype=np.uint8)
    
    threshold = 10
    
    seed_value = image[seed[0], seed[1]]
    
    output_image = np.zeros_like(image)
    
    #Utilizzo valori ternari per la visita nella parte vicina al pixel in questione
    connectivity = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
    
    queue = []
    queue.append(seed)
    
    while len(queue) > 0:
        current_point = queue.pop(0)
        
        visited[current_point[0], current_point[1]] = 255
        
        output_image[current_point[0], current_point[1]] = image[current_point[0], current_point[1]]
        
        for dx, dy in connectivity:
            x = current_point[0] + dx
            y = current_point[1] + dy
            
            if x >= 0 and x < height and y >= 0 and y < width:
                if abs(int(image[x, y]) - int(seed_value)) < threshold and visited[x, y] == 0:
                    queue.append((x, y))
                    visited[x, y] = 255
    
    return output_image

image_path = 'path_to_image.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

seed_point = (100, 100)

output = region_growing(image, seed_point)

cv2.imshow('Original Image', image)
cv2.imshow('Region Growing Output', output)
cv2.waitKey(0)
cv2.destroyAllWindows()
