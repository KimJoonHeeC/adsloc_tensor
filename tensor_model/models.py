from django.db import models
from django.contrib.postgres.fields import JSONField

# Create your models here.

class TensorData(models.Model):
    created_at = models.DateTimeField(blank=True, null=True)
    beacon_name = models.CharField(max_length=255, blank=True, null=True)
    rssis = JSONField(default=dict)