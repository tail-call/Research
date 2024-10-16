    # @!group Plist serialization
    #-------------------------------------------------------------------------#

    # Creates a new object from the given UUID and `objects` hash (of a plist).
    #
    # The method sets up any relationship of the new object, generating the
    # destination object(s) if not already present in the project.
    #
    # @note   This method is used to generate the root object
    #         from a plist. Subsequent invocation are called by the
    #         {AbstractObject#configure_with_plist}. Clients of {Xcodeproj} are
    #         not expected to call this method.
    #
    # @param  [String] uuid
    #         The UUID of the object that needs to be generated.
    #
    # @param  [Hash {String => Hash}] objects_by_uuid_plist
    #         The `objects` hash of the plist representation of the project.
    #
    # @param  [Boolean] root_object
    #         Whether the requested object is the root object and needs to be
    #         retained by the project before configuration to add it to the
    #         `objects` hash and avoid infinite loops.
    #
    # @return [AbstractObject] the new object.
    #
    # @visibility private.
    #
    def new_from_plist(uuid, objects_by_uuid_plist, root_object = false)
      attributes = objects_by_uuid_plist[uuid]
      if attributes
        if Object.const_defined?(attributes['isa'])
          klass = Object.const_get(attributes['isa'])
          object = klass.new(self, uuid)
          objects_by_uuid[uuid] = object
          object.add_referrer(self) if root_object
          object.configure_with_plist(objects_by_uuid_plist)
          object
        end
      end
    end

