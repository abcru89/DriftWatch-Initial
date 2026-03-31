targetScope = 'resourceGroup'

@description('Deployment location. Default is the resource group location.')
param location string = resourceGroup().location

@description('Prefix used to name shared resources.')
param prefix string = 'driftwatch-dev'

@description('ACR name must be globally unique, lowercase, 5-50 chars.')
param acrName string

@description('Storage name must be globally unique, lowercase, 3-24 chars.')
param storageName string

@description('Key Vault name must be globally unique, 3-24 chars.')
param keyVaultName string

param logAnalyticsName string = '${prefix}-law'
param acaEnvName string = '${prefix}-cae'
param identityName string = '${prefix}-uami'
param artifactsContainerName string = 'artifacts'

resource law 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: logAnalyticsName
  location: location
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
  }
}

resource cae 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: acaEnvName
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: law.properties.customerId
        sharedKey: listKeys(law.id, law.apiVersion).primarySharedKey
      }
    }
  }
}

resource uami 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: identityName
  location: location
}

resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: acrName
  location: location
  sku: { name: 'Basic' }
  properties: { adminUserEnabled: false }
}

resource stg 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: storageName
  location: location
  kind: 'StorageV2'
  sku: { name: 'Standard_LRS' }
  properties: {
    minimumTlsVersion: 'TLS1_2'
    allowBlobPublicAccess: false
  }
}

resource artifacts 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-01-01' = {
  name: '${stg.name}/default/${artifactsContainerName}'
  properties: { publicAccess: 'None' }
}

resource kv 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: keyVaultName
  location: location
  properties: {
    tenantId: tenant().tenantId
    enableRbacAuthorization: true
    sku: { family: 'A', name: 'standard' }
    accessPolicies: []
  }
}

var acrPullRole = subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '7f951dda-4ed3-4680-a7ca-43fe172d538d')
var blobDataContributorRole = subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'ba92f5b4-2d11-453d-a403-e96b0029c9fe')
var keyVaultSecretsUserRole = subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '4633458b-17de-408a-b874-0445c86b69e6')

resource raAcrPull 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(resourceGroup().id, acr.id, uami.id, 'AcrPull')
  scope: acr
  properties: {
    roleDefinitionId: acrPullRole
    principalId: uami.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

resource raBlob 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(resourceGroup().id, stg.id, uami.id, 'BlobDataContributor')
  scope: stg
  properties: {
    roleDefinitionId: blobDataContributorRole
    principalId: uami.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

resource raKv 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(resourceGroup().id, kv.id, uami.id, 'KeyVaultSecretsUser')
  scope: kv
  properties: {
    roleDefinitionId: keyVaultSecretsUserRole
    principalId: uami.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

output acaEnvNameOut string = cae.name
output acrLoginServer string = acr.properties.loginServer
output identityNameOut string = uami.name
output storageAccountName string = stg.name
output keyVaultNameOut string = kv.name
