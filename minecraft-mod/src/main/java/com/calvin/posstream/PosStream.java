package com.calvin.posstream;

import net.fabricmc.api.ModInitializer;
import net.fabricmc.fabric.api.itemgroup.v1.ItemGroupEvents;
import net.minecraft.core.Registry;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.core.registries.Registries;
import net.minecraft.resources.Identifier;
import net.minecraft.resources.ResourceKey;
import net.minecraft.world.item.CreativeModeTabs;
import net.minecraft.world.item.Item;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class PosStream implements ModInitializer {
	public static final String MOD_ID = "posstream";

	private static final Logger LOGGER = LoggerFactory.getLogger(MOD_ID);

	public static final ResourceKey<Item> STRUCTURE_SCANNER_KEY = ResourceKey.create(
			Registries.ITEM, Identifier.fromNamespaceAndPath(MOD_ID, "structure_scanner"));

	public static final Item STRUCTURE_SCANNER = new StructureScannerItem(
			new Item.Properties().setId(STRUCTURE_SCANNER_KEY).stacksTo(1));

	@Override
	public void onInitialize() {
		Registry.register(BuiltInRegistries.ITEM, STRUCTURE_SCANNER_KEY, STRUCTURE_SCANNER);

		ItemGroupEvents.modifyEntriesEvent(CreativeModeTabs.TOOLS_AND_UTILITIES)
				.register(entries -> entries.accept(STRUCTURE_SCANNER));

		LOGGER.info("Structure target: {}:{}", StructureScannerItem.HOST, StructureScannerItem.PORT);
	}
}
