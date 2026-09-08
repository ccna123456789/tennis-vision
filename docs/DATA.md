# Données

## Vidéo d'entrée

Le dépôt ne contient aucune vidéo. Les résultats du README ont été obtenus sur un extrait de
retransmission ATP, dont la rediffusion n'est pas libre de droits.

Placez votre propre vidéo dans `data/input/` et renseignez son chemin dans
`configs/default.yaml` :

```yaml
video: data/input/votre_video.mp4
```

### Ce que la vidéo doit contenir

Le pipeline suppose une prise de vue **fixe**, en hauteur, derrière le fond de court, montrant le
court de double en entier — l'angle standard des retransmissions.

Si la caméra bouge, l'homographie calculée sur la première frame devient fausse pour les
suivantes. C'est la limite principale documentée dans le README.

### Calibration

`configs/calibration.json` n'est pas versionné : les 4 coins dépendent de la vidéo et de l'angle
de caméra. Générez le vôtre avec :

```bash
python -m tennisvision calibrate
```

## Modèles

Aucun poids n'est versionné. `yolov8n.pt` (6 Mo) et `yolov8x.pt` (131 Mo) sont téléchargés
automatiquement par Ultralytics au premier lancement.

## Dataset pour entraîner un détecteur de balle

La classe COCO `sports ball` est le point faible du pipeline. Pour entraîner un détecteur dédié :

- **[viren-dhanwani/tennis-ball-detection](https://universe.roboflow.com/viren-dhanwani/tennis-ball-detection)** sur Roboflow
- Licence **CC BY 4.0** — l'attribution du dataset est obligatoire en cas d'utilisation
- Format YOLO, une seule classe `tennis ball`
