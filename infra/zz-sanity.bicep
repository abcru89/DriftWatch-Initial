targetScope = 'resourceGroup'
param location string = resourceGroup().location
output loc string = location
