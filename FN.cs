struct TlsExtension {
    type: ExtensionType
    signatureScheme: SignatureScheme?
    namedGroup: NamedGroup?
}

// Данные

let signatureAlgorithmsExtension = TlsExtension(
    type: ExtensionType.SignatureAlgorithms,
    signatureScheme: SignatureScheme.RSAPkcs1SHA256,

)

let supportedGroupsExtension = TlsExtension(
    type: ExtensionType.SupportedGroups,
    namedGroup: NamedGroup.X25519,
)

public static class Encoder {
    public static void EncodeExtension(TlsExtension extension) {
        Serializer.WriteUInt16(stream, (ushort)extension.type);

        Serializer.WriteUInt16(stream, 4);

        Serializer.WriteUInt16(stream, 2);

        if (extension.signatureScheme != null) {
            Serializer.WriteUInt16(stream, (ushort)extension.signatureScheme);
        }

        if (extension.namedGroup != null) {
            Serializer.WriteUInt16(stream, (ushort)extension.namedGroup);
        }
    }
}