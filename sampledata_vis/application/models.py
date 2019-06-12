from django.db import models
from .validators import validate_file_extension


# Create your models here.
class Data(models.Model):
    title = models.CharField(max_length=100)
    type = models.CharField(max_length=100)
    CSV = models.FileField(upload_to='data/', validators=[validate_file_extension])
    Deletable = models.CharField(max_length=100)

    def __str__(self):
        return self.title

    def delete(self, *args, **kwargs):
        self.CSV.delete()
        super().delete(*args, **kwargs)


