import pygame
import torch


canvas = pygame.Surface((366, 366))
canvas.fill((255,255,255))

pygame.init()
pygame.display.init()
window_size = 366
window = pygame.display.set_mode((window_size, window_size))
clock = pygame.time.Clock()

path = torch.load('Tbest.pt').cpu().long().numpy() # [x1,x2,x3, y1,y2,y3]
NP = int(len(path)/2) #3

Obs_Segments = torch.load('Generate_Obstacle_Segments/Obstacle_Segments.pt')
O = Obs_Segments.shape[0] // 4  # 障碍物数量
Grouped_Obs_Segments = Obs_Segments.reshape(O, 4, 2, 2)  # 注意Grouped_Obs_Segments 和 Obs_Segments 是联动的


while True:
    for p in range(NP-1):
        pygame.draw.line(
            canvas,
            (0, 255, 255),
            (path[p], path[p+NP]),
            (path[p+1], path[p+NP+1]),
            width=4)

        for _ in range(Grouped_Obs_Segments.shape[0]):
            obs_color = (50, 50, 50) if _ < (O - 2) else (225, 100, 0)
            pygame.draw.polygon(canvas, obs_color, Grouped_Obs_Segments[_, :, 0, :].cpu().int().numpy())



    window.blit(canvas, canvas.get_rect())
    pygame.event.pump()
    pygame.display.update()
    clock.tick(1)