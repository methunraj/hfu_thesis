from PIL.ExifTags import TAGS, GPSTAGS

def get_geolocation(image):
    def get_geotagging(exif):
        if not exif:
            return {}

        geotagging = {}
        for (idx, tag) in TAGS.items():
            if tag == 'GPSInfo':
                if idx not in exif:
                    return {}

                for (t, value) in GPSTAGS.items():
                    if t in exif[idx]:
                        geotagging[value] = exif[idx][t]

        return geotagging

    def get_decimal_from_dms(dms, ref):
        degrees = dms[0]
        minutes = dms[1] / 60.0
        seconds = dms[2] / 3600.0

        if ref in ['S', 'W']:
            degrees = -degrees
            minutes = -minutes
            seconds = -seconds

        return round(degrees + minutes + seconds, 5)

    exif = image._getexif()
    geotagging = get_geotagging(exif)
    if 'GPSLatitude' in geotagging and 'GPSLongitude' in geotagging:
        lat = get_decimal_from_dms(geotagging['GPSLatitude'], geotagging['GPSLatitudeRef'])
        lon = get_decimal_from_dms(geotagging['GPSLongitude'], geotagging['GPSLongitudeRef'])
    else:
        lat, lon = None, None

    return lat, lon
