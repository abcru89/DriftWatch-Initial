param location string = 'eastus2'

param acaEnvName string
param acrName string
param identityName string
param appName string
param image string

param aadClientId string
@secure()
param aadClientSecret string

@secure()
param authSigningSecret string
@secure()
param authEncryptionSecret string

param targetPort int = 8501
param minReplicas int = 1
param maxReplicas int = 2
param cpu string = '0.5'
param memory string = '1Gi'

resource cae 'Microsoft.App/managedEnvironments@2024-03-01' existing = {
  name: acaEnvName
}

resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' existing = {
  name: acrName
}

resource uami 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' existing = {
  name: identityName
}

var issuer = '${environment().authentication.loginEndpoint}${tenant().tenantId}/v2.0'

resource app 'Microsoft.App/containerApps@2024-03-01' = {
  name: appName
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${uami.id}': {}
    }
  }
  properties: {
    managedEnvironmentId: cae.id
    configuration: {
      secrets: [
        { name: 'aad-client-secret', value: aadClientSecret }
        { name: 'auth-signing-secret', value: authSigningSecret }
        { name: 'auth-encryption-secret', value: authEncryptionSecret }
      ]
      ingress: {
        external: true
        targetPort: targetPort
        transport: 'auto'
        allowInsecure: false
      }
      registries: [
        { server: acr.properties.loginServer, identity: uami.id }
      ]
    }
    template: {
      containers: [
        {
          name: 'driftwatch'
          image: image
          resources: {
            cpu: json(cpu)
            memory: memory
          }
        }
      ]
      scale: { minReplicas: minReplicas, maxReplicas: maxReplicas }
    }
  }
}

resource auth 'Microsoft.App/containerApps/authConfigs@2024-03-01' = {
  parent: app
  name: 'current'
  properties: {
    platform: { enabled: true }
    globalValidation: {
      redirectToProvider: 'azureactivedirectory'
      unauthenticatedClientAction: 'RedirectToLoginPage'
    }
    httpSettings: { requireHttps: true }
    encryptionSettings: {
      containerAppAuthEncryptionSecretName: 'auth-encryption-secret'
      containerAppAuthSigningSecretName: 'auth-signing-secret'
    }
    identityProviders: {
      azureActiveDirectory: {
        enabled: true
        registration: {
          clientId: aadClientId
          clientSecretSettingName: 'aad-client-secret'
          openIdIssuer: issuer
        }
      }
    }
    login: {
      tokenStore: { enabled: false }
    }
  }
}

output url string = 'https://${app.properties.configuration.ingress.fqdn}'
